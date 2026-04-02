import asyncio
import websockets
import numpy as np
import json
import pathlib
import os
import logging
import math
import imageio
import random
from dataclasses import dataclass, field
from typing import List, Tuple

CUSTOM_LOG_DIR = "/mnt/data/sftp/data/quangpt3/Evo-1/Evo_1/logs"
os.makedirs(CUSTOM_LOG_DIR, exist_ok=True)

_original_file_handler = logging.FileHandler

def _patched_file_handler(filename, mode='a', encoding=None, delay=False):
    if filename == "/tmp/robosuite.log":
        filename = os.path.join(CUSTOM_LOG_DIR, "robosuite.log")
    return _original_file_handler(filename, mode, encoding, delay)

logging.FileHandler = _patched_file_handler


from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["MUJOCO_GL"] = "osmes"
LIBERO_DUMMY_ACTION = [0.0] * 6 + [0.0]

@dataclass
class Config:
    horizon: int = 14
    max_steps: List[int] = field(default_factory=lambda: [25, 25, 25, 95])
    server_url: str = "ws://0.0.0.0:9000"
    ckpt_name: str = "Evo1_libero_all"
    task_suites: List[str] = field(default_factory=lambda: ["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    log_dir: str = "./log_file"
    video_dir: str = "./video_log_file"
    num_episodes: int = 1
    seed: int = 42
    camera_res: int = 448

    @property
    def log_file(self) -> str:
        return os.path.join(self.log_dir, f"{self.ckpt_name}.txt")

def setup_logger(config: Config) -> logging.Logger:
    os.makedirs(config.log_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(config.log_file, mode='a')
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

@dataclass
class EpisodeResult:
    success: bool
    steps: int
    frames: List[np.ndarray]

class LiberoEnvAdapter:
    def __init__(self, task, config: Config):
        self.config = config
        self.env, self.task_description = self._setup_env(task)
        self._dummy_proc = np.zeros((config.camera_res, config.camera_res, 3), dtype=np.uint8).tolist()

    def _setup_env(self, task):
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": self.config.camera_res, "camera_widths": self.config.camera_res}
        env = OffScreenRenderEnv(**env_args)
        env.seed(self.config.seed)
        return env, task.language

    def _quat2axisangle(self, quat):
        quat[3] = np.clip(quat[3], -1.0, 1.0)
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def _extract_state(self, obs):
        return np.concatenate((
            obs["robot0_eef_pos"],
            self._quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )).tolist()

    def get_json_payload(self, obs) -> str:
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8).tolist()
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8).tolist()
        
        data = {
            "image": [img, wrist_img, self._dummy_proc],
            "state": self._extract_state(obs),
            "prompt": self.task_description,
            "image_mask": [1, 1, 0],
            "action_mask": [1] * 7 + [0] * 17,
        }
        return json.dumps(data)

    def reset(self):
        self.env.reset()

    def set_init_state(self, state):
        return self.env.set_init_state(state)

    def step(self, action):
        return self.env.step(action)

class Evaluator:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    async def _save_video_async(self, frames, filename, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        if frames:
            import functools
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, functools.partial(imageio.mimsave, filepath, frames, fps=30))

    async def run_suite(self, task_suite_name: str, max_steps: int):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        
        total_success, total_episodes, total_steps = 0, 0, 0

        async with websockets.connect(self.config.server_url) as ws:
            self.logger.info(f"========== Start task suite {task_suite_name} ==========")

            for task_id in range(task_suite.n_tasks):
                task = task_suite.get_task(task_id)
                initial_states = task_suite.get_task_init_states(task_id)
                adapter = LiberoEnvAdapter(task, self.config)

                success_count, ep_count, steps_count = await self._run_single_task(
                    task_id=task_id,
                    task_suite_name=task_suite_name,
                    adapter=adapter,
                    initial_states=initial_states,
                    ws=ws,
                    max_steps=max_steps
                )
                
                total_success += success_count
                total_episodes += ep_count
                total_steps += steps_count

            self.logger.info("\n========= Overall Task Summary =========")
            self.logger.info(f"Total Successful: {total_success}/{total_episodes}")
            if total_episodes > 0:
                self.logger.info(f"Average Steps: {total_steps / total_episodes:.2f}")

    async def _run_single_task(self, task_id: int, task_suite_name: str, adapter: LiberoEnvAdapter, initial_states: list, ws, max_steps: int) -> Tuple[int, int, int]:
        self.logger.info(f"\n========= Start task {task_id+1}: {adapter.task_description} =========")
        
        task_success = 0
        total_steps = 0
        task_episodes = min(self.config.num_episodes, len(initial_states))

        for ep in range(task_episodes):
            result = await self._run_single_episode(adapter, initial_states[ep], ws, max_steps)
            
            video_dir = os.path.join(self.config.video_dir, self.config.ckpt_name, task_suite_name)
            await self._save_video_async(result.frames, f"task{task_id+1}_episode{ep+1}.mp4", video_dir)

            if result.success:
                task_success += 1
                total_steps += result.steps
                self.logger.info(f"Task {task_id} | Episode {ep+1}: Success (Steps: {result.steps})")
            else:
                self.logger.info(f"Task {task_id} | Episode {ep+1}: Fail")

        self.logger.info(f"=== Task {task_id + 1} Summary: {task_success}/{task_episodes} Successful ===")
        return task_success, task_episodes, total_steps

    async def _run_single_episode(self, adapter: LiberoEnvAdapter, init_state, ws, max_steps: int) -> EpisodeResult:
        adapter.reset()
        obs = adapter.set_init_state(init_state)
        
        for _ in range(10):
            obs, _, _, _ = adapter.step(LIBERO_DUMMY_ACTION)

        frames = []
        actual_steps = 0

        for step in range(max_steps):
            actual_steps += 1
            
            await ws.send(adapter.get_json_payload(obs))

            try:
                result = await ws.recv()
                actions = np.array(json.loads(result))
            except Exception as e:
                self.logger.error(f"Failed to parse action: {e}")
                break

            episode_done, obs = self._execute_horizon_actions(adapter, actions, frames, obs)
            
            if episode_done:
                return EpisodeResult(success=True, steps=actual_steps, frames=frames)

        return EpisodeResult(success=False, steps=actual_steps, frames=frames)

    def _execute_horizon_actions(self, adapter: LiberoEnvAdapter, actions: np.ndarray, frames: list, current_obs) -> Tuple[bool, dict]:
        obs = current_obs
        for i in range(self.config.horizon):
            action = actions[i].tolist()
            action[6] = -1 if action[6] > 0.5 else 1
            
            try:
                obs, _, done, _ = adapter.step(action[:7])
            except ValueError as e:
                self.logger.warning(f"Invalid action encountered: {e}")
                return False, obs
            
            frame = np.hstack([
                np.rot90(obs["agentview_image"], 2),
                np.rot90(obs["robot0_eye_in_hand_image"], 2)
            ])
            frames.append(frame)

            if done:
                return True, obs
                
        return False, obs

async def main():
    config = Config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    logger = setup_logger(config)
    evaluator = Evaluator(config, logger)
    
    for name, max_steps in zip(config.task_suites, config.max_steps):
        await evaluator.run_suite(name, max_steps)

if __name__ == "__main__":
    asyncio.run(main())