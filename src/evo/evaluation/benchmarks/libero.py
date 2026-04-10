import asyncio
import websockets
import numpy as np
import json
import os
import logging
import math
import imageio
import random
from pathlib import Path
import yaml
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from dataclasses import dataclass

os.environ["MUJOCO_GL"] = "osmes"
LIBERO_DUMMY_ACTION = [0.0] * 6 + [0.0]

@dataclass
class EpisodeResult:
    success: bool
    steps: int
    frames: list

def encode_image_array(img_array: np.ndarray):
    return img_array.astype(np.uint8).tolist()

def quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def obs_to_json_dict(obs, prompt, resize_size=448):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    dummy_proc = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)

    return {
        "image": [
            encode_image_array(img),
            encode_image_array(wrist_img),
            encode_image_array(dummy_proc)
        ],
        "state": np.concatenate((
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )).tolist(),
        "prompt": prompt,
        "image_mask": [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17,
    }

class LiberoEnvAdapter:
    def __init__(self, task, resolution=448, seed=42):
        self.task_description = task.language
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self.env = OffScreenRenderEnv(bddl_file_name=str(task_bddl_file), camera_heights=resolution, camera_widths=resolution)
        self.env.seed(seed)

    def reset(self):
        return self.env.reset()
        
    def set_init_state(self, state):
        return self.env.set_init_state(state)
        
    def step(self, action):
        return self.env.step(action)

def save_video(frames, filename="simulation.mp4", fps=20, save_dir="videos"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    if frames:
        imageio.mimsave(filepath, frames, fps=fps)

class LiberoBenchmark:
    def __init__(self, horizon: int, camera_res: int, seed: int, num_episodes: int, video_dir: str, ckpt_name: str, logger: logging.Logger, debug=False, start_task_idx=0):
        self.horizon = horizon
        self.camera_res = camera_res
        self.seed = seed
        self.num_episodes = num_episodes
        self.video_dir = video_dir
        self.ckpt_name = ckpt_name
        self.logger = logger
        self.debug = debug
        self.start_task_idx = start_task_idx

    async def run_suite(self, task_suite_name: str, max_steps: int, ws):
        suite = benchmark.get_benchmark_dict()[task_suite_name]()
        self.logger.info(f"Starting task suite {task_suite_name} ({suite.n_tasks} tasks)")
        
        total_success, total_episodes, total_steps = 0, 0, 0
        if self.debug:
            suite.n_tasks = 1
        
        for task_id in range(suite.n_tasks):
            if task_id < self.start_task_idx:
                self.logger.info(f"Skipping Task {task_id + 1}...")
                continue
                
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            
            adapter = LiberoEnvAdapter(task, resolution=self.camera_res, seed=self.seed)
            task_description = adapter.task_description
            env = adapter.env

            task_episodes = min(self.num_episodes, len(initial_states))
            task_success = 0
            
            for ep in range(task_episodes):
                env.reset()
                obs = env.set_init_state(initial_states[ep])
                
                for _ in range(10): obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                
                prompt = str(task_description)
                frames = []
                episode_done = False
                steps_taken = 0
                
                for step in range(max_steps):
                    steps_taken += 1
                    await ws.send(json.dumps(obs_to_json_dict(obs, prompt, resize_size=self.camera_res)))
                    
                    try:
                        actions = np.array(json.loads(await ws.recv()))
                    except Exception as e:
                        self.logger.error(f"Action parsing failed: {e}")
                        break

                    for i in range(self.horizon):
                        action = actions[i].tolist()
                        action[6] = -1 if action[6] > 0.5 else 1
                        
                        try:
                            obs, reward, done, _ = env.step(action[:7])
                        except ValueError as ve:
                            self.logger.error(f"Invalid action: {ve}")
                            break
                            
                        frames.append(np.hstack([np.rot90(obs["agentview_image"], 2), np.rot90(obs["robot0_eye_in_hand_image"], 2)]))
                        
                        if done or reward > 0:
                            episode_done = True
                            task_success += 1
                            total_success += 1
                            total_steps += steps_taken
                            break
                            
                    if episode_done: break

                vid_dir = f"{self.video_dir}/{self.ckpt_name}/{task_suite_name}"
                save_video(frames, f"task{task_id+1}_episode{ep+1}.mp4", fps=20, save_dir=vid_dir)
                self.logger.info(f"Task {task_id} | Ep {ep+1}: {'Success' if episode_done else 'Fail'}")

            self.logger.info(f"Task {task_id + 1} Summary: {task_success}/{task_episodes} Successful")
            total_episodes += task_episodes

        if total_episodes > 0:
            self.logger.info(f"Overall {task_suite_name}: {total_success}/{total_episodes} Success | Avg Steps: {total_steps/total_episodes:.2f}")

