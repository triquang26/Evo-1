import os
import asyncio
import websockets
import numpy as np
import cv2
import json
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import yaml
import argparse

from src.evo.models.evo1 import EVO1

class Normalizer:
    def __init__(self, stats_or_path):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        def pad_to_24(x):
            x = torch.tensor(x, dtype=torch.float32)
            if x.shape[0] < 24:
                pad = torch.zeros(24 - x.shape[0], dtype=torch.float32)
                x = torch.cat([x, pad], dim=0)
            elif x.shape[0] > 24:
                raise ValueError(f"Input length {x.shape[0]} exceeds expected 24")
            return x

        if len(stats) != 1:
            raise ValueError(f"norm_stats.json should contain only one robot key, but: {list(stats.keys())}")

        robot_key = list(stats.keys())[0]
        robot_stats = stats[robot_key]

        self.state_min = pad_to_24(robot_stats["observation.state"]["min"])
        self.state_max = pad_to_24(robot_stats["observation.state"]["max"])
        self.action_min = pad_to_24(robot_stats["action"]["min"])
        self.action_max = pad_to_24(robot_stats["action"]["max"])

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state_min = self.state_min.to(state.device, dtype=state.dtype)
        state_max = self.state_max.to(state.device, dtype=state.dtype)
        return torch.clamp(2 * (state - state_min) / (state_max - state_min + 1e-8) - 1, -1.0, 1.0)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action_min = self.action_min.to(action.device, dtype=action.dtype)
        action_max = self.action_max.to(action.device, dtype=action.dtype)
        if action.ndim == 1:
            action = action.view(1, -1)
        return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min

def load_server_config(config_path="configs/serving/server.yaml"):
    project_root = Path(__file__).resolve().parents[3]
    path = project_root / config_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_model_and_normalizer(ckpt_dir, timesteps=32, device="cuda"):
    config_path = os.path.join(ckpt_dir, "config.json")
    stats_path = os.path.join(ckpt_dir, "norm_stats.json")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    with open(stats_path, "r") as f:
        stats = json.load(f)

    model_cfg = config.get("model", config)
    if "train" in config:
        config["train"]["finetune_vlm"] = False
        config["train"]["finetune_action_head"] = False
    else:
        config["finetune_vlm"] = False
        config["finetune_action_head"] = False
        
    if "flowmatching" in model_cfg:
        model_cfg["flowmatching"]["num_inference_timesteps"] = timesteps
    else:
        config["num_inference_timesteps"] = timesteps

    model = EVO1(config).eval()
    ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint.get("module", checkpoint), strict=False)
    except FileNotFoundError:
        print(f"[Warning] No PyTorch ckpt found at {ckpt_path}.")
    
    model = model.to(device)
    normalizer = Normalizer(stats)
    return model, normalizer

def decode_image_from_list(img_list, device="cuda"):
    img_array = np.array(img_list, dtype=np.uint8)
    img = cv2.resize(img_array, (448, 448))
    pil = Image.fromarray(img)
    return transforms.ToTensor()(pil).to(device)

def infer_from_json_dict(data: dict, model, normalizer, device="cuda"):
    images = [decode_image_from_list(img, device) for img in data["image"]]
    if len(images) != 3:
        raise ValueError(f"Must provide exactly 3 images. Got {len(images)}")
    
    state = torch.tensor(data["state"], dtype=torch.float32, device=device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[1] < 24:
        state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device=device)], dim=1)
        
    norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)

    prompt = data["prompt"]
    image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device=device)
    action_mask = torch.tensor([data["action_mask"]], dtype=torch.int32, device=device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask
        )
        action = action.reshape(1, -1, 24)
        action = normalizer.denormalize_action(action[0])
        return action.cpu().numpy().tolist()

async def handle_request(websocket, model, normalizer, device="cuda"):
    print("Client connected")
    try:
        async for message in websocket:
            json_data = json.loads(message)
            actions = infer_from_json_dict(json_data, model, normalizer, device)
            await websocket.send(json.dumps(actions))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/serving/server.yaml")
    args = parser.parse_args()
    
    cfg = load_server_config(args.config)
    
    port = cfg.get("port", 9000)
    host = cfg.get("host", "0.0.0.0")
    ckpt_dir = cfg.get("checkpoint_dir")
    device = cfg.get("device", "cuda")
    
    if not ckpt_dir or not os.path.exists(ckpt_dir):
        raise ValueError(f"Invalid checkpoint_dir: {ckpt_dir}. Please set a valid path in {args.config}")
        
    print(f"Loading EVO_1 model from {ckpt_dir}...")
    model, normalizer = load_model_and_normalizer(ckpt_dir, timesteps=cfg.get("num_inference_timesteps", 32), device=device)

    async def main():
        print(f"EVO_1 server running at ws://{host}:{port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer, device),
            host, port, max_size=cfg.get("max_message_size", 100_000_000)
        ):
            await asyncio.Future()

    asyncio.run(main())
