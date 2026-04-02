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

os.environ["MUJOCO_GL"] = "osmes"
LIBERO_DUMMY_ACTION = [0.0] * 6 + [0.0]

def load_config(config_path="configs/eval/libero.yaml"):
    project_root = Path(__file__).resolve().parents[4]
    path = project_root / config_path
    with open(path, 'r') as f:
        return yaml.safe_load(f)

cfg = load_config()

os.makedirs(cfg['log_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{cfg['log_dir']}/{cfg['ckpt_name']}.txt", mode='a'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

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

def get_libero_env(task, resolution=448, seed=42):
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(task_bddl_file), camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task_description

def save_video(frames, filename="simulation.mp4", fps=20, save_dir="videos"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    if frames:
        imageio.mimsave(filepath, frames, fps=fps)
        log.info(f"Video saved: {filepath} ({len(frames)} frames)")

async def run_evaluation(task_suite_name: str, max_steps: int):
    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    log.info(f"Starting task suite {task_suite_name} ({suite.n_tasks} tasks)")
    
    total_success, total_episodes, total_steps = 0, 0, 0

    async with websockets.connect(cfg['server_url']) as ws:
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, seed=cfg['seed'])

            task_episodes = min(cfg['num_episodes'], len(initial_states))
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
                    await ws.send(json.dumps(obs_to_json_dict(obs, prompt)))
                    
                    try:
                        actions = np.array(json.loads(await ws.recv()))
                    except Exception as e:
                        log.error(f"Action parsing failed: {e}")
                        break

                    for i in range(cfg['horizon']):
                        action = actions[i].tolist()
                        action[6] = -1 if action[6] > 0.5 else 1
                        
                        try:
                            obs, reward, done, _ = env.step(action[:7])
                        except ValueError as ve:
                            log.error(f"Invalid action: {ve}")
                            break
                            
                        frames.append(np.hstack([np.rot90(obs["agentview_image"], 2), np.rot90(obs["robot0_eye_in_hand_image"], 2)]))
                        
                        if done:
                            episode_done = True
                            task_success += 1
                            total_success += 1
                            total_steps += steps_taken
                            break
                            
                    if episode_done: break

                vid_dir = f"{cfg['video_dir']}/{cfg['ckpt_name']}/{task_suite_name}"
                save_video(frames, f"task{task_id+1}_episode{ep+1}.mp4", fps=cfg['video_fps'], save_dir=vid_dir)
                log.info(f"Task {task_id} | Ep {ep+1}: {'Success' if episode_done else 'Fail'}")

            log.info(f"Task {task_id + 1} Summary: {task_success}/{task_episodes} Successful")
            total_episodes += task_episodes

    if total_episodes > 0:
        log.info(f"Overall {task_suite_name}: {total_success}/{total_episodes} Success | Avg Steps: {total_steps/total_episodes:.2f}")

if __name__ == "__main__":
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    
    for name, max_steps in zip(cfg['task_suites'], cfg['max_steps']):
        asyncio.run(run_evaluation(name, max_steps))
