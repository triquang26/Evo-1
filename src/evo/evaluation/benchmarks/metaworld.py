import asyncio
import json
import os
import cv2
import gymnasium as gym
import metaworld
import numpy as np
import websockets
import datetime
from pathlib import Path
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")
gym.logger.min_level = gym.logger.ERROR

def load_config(config_path="configs/eval/metaworld.yaml"):
    project_root = Path(__file__).resolve().parents[4]
    path = project_root / config_path
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['project_root'] = str(project_root)
    return cfg

cfg = load_config()
os.makedirs(cfg['log_dir'], exist_ok=True)
log_path = os.path.join(cfg['log_dir'], f"mt50_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def log_write(text: str):
    print(text)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def encode_image(img_bgr: np.ndarray):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8).tolist()

def obs_to_state(obs, take=8):
    if isinstance(obs, dict):
        arr = np.asarray(obs["observation"]).ravel() if "observation" in obs else np.concatenate([np.asarray(v).ravel() for v in obs.values()])
    else:
        arr = np.asarray(obs).ravel()
    return arr[:min(take, arr.shape[0])].tolist()

def render_env(env):
    rgb = np.ascontiguousarray(env.render(), dtype=np.uint8)
    rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    rgb = cv2.resize(rgb, tuple(cfg['img_size']), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)

async def infer_action(ws, img_bgr, state_vec, prompt):
    dummy_img = np.zeros((448, 448, 3), dtype=np.uint8)
    payload = {
        "image": [encode_image(img_bgr), encode_image(dummy_img), encode_image(dummy_img)],
        "state": state_vec,
        "prompt": prompt,              
        "image_mask": [1, 0, 0],
        "action_mask": [1, 1, 1, 1] + [0]*20,
    }
    await ws.send(json.dumps(payload))
    return np.asarray(json.loads(await ws.recv()), dtype=np.float32)

def load_mt50_data():
    order_path = os.path.join(cfg['project_root'], cfg['order_json_path'])
    tasks_path = os.path.join(cfg['project_root'], cfg['tasks_jsonl_path'])
    
    with open(order_path, "r") as f:
        data = json.load(f)
        
    ordered_indices = [int(x) for x in data["ordered_indices"]]
    groups = {k: set(v) for k, v in data["groups"].items()}
    idx_to_slug = {int(k): v for k, v in data["idx_to_slug"].items()}
    
    prompts = {}
    if os.path.exists(tasks_path):
        with open(tasks_path, "r") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if "idx" in obj: prompts[int(obj["idx"])] = obj["task"]
                    elif "slug" in obj: prompts[obj["slug"]] = obj["task"]

    return ordered_indices, groups, idx_to_slug, prompts

async def evaluate():
    envs = gym.make_vec("Meta-World/MT50", vector_strategy="sync", seed=cfg['seed'], render_mode="rgb_array", camera_name=cfg['camera_name'])
    indices, groups, idx_to_slug, prompts = load_mt50_data()
    
    if cfg['target_level'].lower() != "all":
        allowed = groups.get(cfg['target_level'].lower(), set())
        indices = [i for i in indices if idx_to_slug.get(i) in allowed]
        
    success_counts, trials_counts = {i: 0 for i in indices}, {i: 0 for i in indices}
    group_success = {g: 0 for g in ["easy", "medium", "hard", "very_hard"]}
    group_trials = {g: 0 for g in ["easy", "medium", "hard", "very_hard"]}

    os.makedirs(cfg['video_save_dir'], exist_ok=True)

    async with websockets.connect(cfg['server_url'], max_size=100000000) as ws:
        for idx in indices:
            sub = envs.envs[idx]
            slug = idx_to_slug.get(idx, f"task-{idx}")
            prompt = prompts.get(idx, prompts.get(slug, ""))
            gname = next((g for g in group_trials if slug in groups.get(g, set())), None)

            for ep in range(cfg['episodes']):
                for obj in (sub, getattr(sub, "unwrapped", None)):
                    if hasattr(obj, "iterate_goal_position"):
                        try: obj.iterate_goal_position()
                        except: pass
                        break

                obs, _ = sub.reset(seed=cfg['seed'] + ep)
                trials_counts[idx] += 1
                if gname: group_trials[gname] += 1

                video_path = os.path.join(cfg['video_save_dir'], f"task{idx:02d}_{slug}_ep{ep+1:03d}.mp4")
                video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), cfg['video_fps'], tuple(cfg['img_size']))

                steps, done = 0, False
                try: obs, _, _, _, _ = sub.step(np.zeros(sub.action_space.shape, dtype=np.float32))
                except: pass

                while steps < cfg['episode_horizon'] and not done:
                    img_bgr = render_env(sub)
                    for _ in range(cfg['video_dup_frames']): video_writer.write(img_bgr)
                    
                    actions = await infer_action(ws, img_bgr, obs_to_state(obs, cfg['state_take']), prompt)
                    
                    for i in range(cfg['horizon']):
                        a4 = np.clip(np.asarray(actions[i][:4], dtype=np.float32), sub.action_space.low, sub.action_space.high)
                        obs, _, term, trunc, info = sub.step(a4)
                        steps += 1
                        
                        if info.get("success", 0) == 1:
                            success_counts[idx] += 1
                            if gname: group_success[gname] += 1
                            done = True
                            break
                        if term or trunc or steps >= cfg['episode_horizon']:
                            done = True
                            break
                            
                video_writer.release()
            log_write(f"[Task {idx} {slug}] success_rate={success_counts[idx]/trials_counts[idx]:.3f}")
    envs.close()
    
    log_write("\n==== Difficulty buckets ====")
    for g in group_success:
        log_write(f"{g:10s} : {group_success[g]/group_trials[g] if group_trials[g] > 0 else 0:.3f}")

if __name__ == "__main__":
    asyncio.run(evaluate())
