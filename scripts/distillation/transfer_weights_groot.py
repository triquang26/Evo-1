import torch
import argparse
import logging
import yaml
import os
import sys
from pathlib import Path
from transformers import AutoModel, AutoConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Dynamically add Isaac-GR00T to python path if not present
project_root = Path(__file__).resolve().parent.parent.parent
isaac_path = str(project_root / "externals" / "Isaac-GR00T")
if isaac_path not in sys.path:
    sys.path.insert(0, isaac_path)

def build_layer_map(indices_list):
    """
    Builds a dictionary mapping from the teacher layout (values in array)
    to the student layout (consecutive indices).
    E.g. [0, 2, 4] -> {0: 0, 2: 1, 4: 2}
    """
    return {teacher_idx: student_idx for student_idx, teacher_idx in enumerate(indices_list)}

def transfer_weights_groot(config_dict, output_path):
    model_cfg = config_dict.get('model', {})
    teacher_cfg = config_dict.get('teacher_cfg', {})
    
    hf_repo_id = teacher_cfg.get('hf_repo_id', "liorbenhorin-nv/groot-libero_10-64_40000")
    
    vlm_indices = model_cfg.get('vlm_layer_indices')
    action_head_indices = model_cfg.get('action_head_layer_indices')
    
    if not vlm_indices or not action_head_indices:
        raise ValueError("Config must contain 'vlm_layer_indices' and 'action_head_layer_indices' under 'model'.")
        
    vlm_map = build_layer_map(vlm_indices)
    action_head_map = build_layer_map(action_head_indices)

    logging.info(f"VLM Layer Map: {vlm_map}")
    logging.info(f"Action Head Layer Map: {action_head_map}")
    logging.info(f"Using Teacher GR00T from HF Repo: {hf_repo_id}...")
    
    logging.info("Downloading and initializing teacher model ...")
    teacher_model = AutoModel.from_pretrained(hf_repo_id, trust_remote_code=True)
    state_dict = teacher_model.state_dict()
    
    new_state_dict = {}
    skipped_keys = []
    mapped_keys = []

    for key, value in state_dict.items():
        # 1. Action Head mapping
        if "action_head.model.transformer_blocks." in key:
            parts = key.split("action_head.model.transformer_blocks.")
            head_part = parts[0]
            tail_part = parts[1]
            layer_idx = int(tail_part.split(".")[0])
            
            if layer_idx in action_head_map:
                new_idx = action_head_map[layer_idx]
                new_key = f"{head_part}action_head.model.transformer_blocks.{new_idx}.{'.'.join(tail_part.split('.')[1:])}"
                new_state_dict[new_key] = value.clone()
                mapped_keys.append(f"Action Head: {key} -> {new_key}")
            else:
                skipped_keys.append(key)
                
        # 2. VLM mapping
        elif "backbone.model.language_model.model.layers." in key:
            parts = key.split("backbone.model.language_model.model.layers.")
            head_part = parts[0]
            tail_part = parts[1]
            layer_idx = int(tail_part.split(".")[0])
            
            if layer_idx in vlm_map:
                new_idx = vlm_map[layer_idx]
                new_key = f"{head_part}backbone.model.language_model.model.layers.{new_idx}.{'.'.join(tail_part.split('.')[1:])}"
                new_state_dict[new_key] = value.clone()
                mapped_keys.append(f"VLM: {key} -> {new_key}")
            else:
                skipped_keys.append(key)
        else:
            # Everything else (projectors, embedders, etc.) copies natively
            new_state_dict[key] = value.clone()
            
    logging.info(f"Summary: Mapped {len(mapped_keys)} specific layer keys.")
    logging.info(f"Summary: Skipped {len(skipped_keys)} unused layer keys.")
    logging.info(f"Total keys in new state dict: {len(new_state_dict)}")
    
    # Save as standard HuggingFace model directory
    mini_config = teacher_model.config
    mini_config.hidden_size = model_cfg.get('hidden_size', 1024)
    if vlm_indices:
        mini_config.select_layer = len(vlm_indices)
    if action_head_indices and hasattr(mini_config, "diffusion_model_cfg"):
        mini_config.diffusion_model_cfg["num_layers"] = len(action_head_indices)
        
    student_model = AutoModel.from_config(mini_config)
    student_model.load_state_dict(new_state_dict, strict=True)
    
    os.makedirs(output_path, exist_ok=True)
    student_model.save_pretrained(output_path)
    logging.info(f"Successfully saved HuggingFace compatible GR00T-Mini to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer weights dynamically via config for GR00T-Mini.")
    parser.add_argument("--config", type=str, required=True, help="Path to your YAML training configuration (e.g. configs/train/distill_groot.yaml)")
    parser.add_argument("--output_ckpt", type=str, required=True, help="Path to output directory for saving the HuggingFace model")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    transfer_weights_groot(cfg, args.output_ckpt)
