import torch
import sys
from pathlib import Path

# Thêm đường dẫn project
project_root = Path("/mnt/data/sftp/data/quangpt3/Evo-1")
sys.path.append(str(project_root))

from src.evo.models.builder import build_model
import json

ckpt_dir = "/mnt/data/sftp/data/quangpt3/Evo-1/checkpoint_distill_stage1_fullfinetuning_v02-2/step_best"

with open(f"{ckpt_dir}/config.json", "r") as f:
    config = json.load(f)

# Mock some properties
config["model"]["type"] = "evo1_student"

print("Building EVO1Student model...")
model = build_model(config)

student_keys = list(model.state_dict().keys())
student_vlm_keys = [k for k in student_keys if "embedder" in k and "layers" in k]
print(f"Student VLM layer keys count: len({student_vlm_keys})")
if student_vlm_keys:
    print(f"Sample student keys: {student_vlm_keys[:5]}")

print("\nLoading checkpoint...")
ckpt_path = f"{ckpt_dir}/mp_rank_00_model_states.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint.get("module", checkpoint)

ckpt_keys = list(state_dict.keys())
ckpt_vlm_keys = [k for k in ckpt_keys if "embedder" in k and "layers" in k]
print(f"Checkpoint VLM layer keys count: len({ckpt_vlm_keys})")
if ckpt_vlm_keys:
    print(f"Sample ckpt keys: {ckpt_vlm_keys[:5]}")

# Check intersection
missing = set(student_keys) - set(ckpt_keys)
unexpected = set(ckpt_keys) - set(student_keys)

print(f"\nMissing keys (expected by model but not in ckpt): {len(missing)}")
if missing:
    print(f"Sample missing keys: {list(missing)[:5]}")

print(f"\nUnexpected keys (in ckpt but not in model): {len(unexpected)}")
if unexpected:
    print(f"Sample unexpected keys: {list(unexpected)[:5]}")
