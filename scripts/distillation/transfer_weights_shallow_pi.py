import torch
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def transfer_weights(teacher_ckpt_path, output_path, action_head_map, vlm_map):
    if not os.path.exists(teacher_ckpt_path):
        raise FileNotFoundError(f"Cannot find teacher checkpoint at {teacher_ckpt_path}")

    logging.info(f"Loading Teacher checkpoint from {teacher_ckpt_path}...")
    state_dict = torch.load(teacher_ckpt_path, map_location='cpu')
    
    is_module = False
    if "module" in state_dict:
        state_dict = state_dict["module"]
        is_module = True

    new_state_dict = {}
    skipped_keys = []
    mapped_keys = []

    for key, value in state_dict.items():
        # Action Head mapping
        if "action_head.transformer_blocks." in key:
            parts = key.split("action_head.transformer_blocks.")
            head_part = parts[0]
            tail_part = parts[1]
            layer_idx_str = tail_part.split(".")[0]
            layer_idx = int(layer_idx_str)
            
            if layer_idx in action_head_map:
                new_idx = action_head_map[layer_idx]
                new_key = f"{head_part}action_head.transformer_blocks.{new_idx}.{'.'.join(tail_part.split('.')[1:])}"
                new_state_dict[new_key] = value.clone()
                mapped_keys.append(f"Action Head: {key} -> {new_key}")
            else:
                skipped_keys.append(key)
                
        # VLM mapping
        elif "embedder.model.language_model.model.layers." in key:
            parts = key.split("embedder.model.language_model.model.layers.")
            head_part = parts[0]
            tail_part = parts[1]
            layer_idx_str = tail_part.split(".")[0]
            layer_idx = int(layer_idx_str)
            
            if layer_idx in vlm_map:
                new_idx = vlm_map[layer_idx]
                new_key = f"{head_part}embedder.model.language_model.model.layers.{new_idx}.{'.'.join(tail_part.split('.')[1:])}"
                new_state_dict[new_key] = value.clone()
                mapped_keys.append(f"VLM: {key} -> {new_key}")
            else:
                skipped_keys.append(key)
        elif "embedder.model.language_model.layers." in key:
            parts = key.split("embedder.model.language_model.layers.")
            head_part = parts[0]
            tail_part = parts[1]
            layer_idx_str = tail_part.split(".")[0]
            layer_idx = int(layer_idx_str)
            
            if layer_idx in vlm_map:
                new_idx = vlm_map[layer_idx]
                new_key = f"{head_part}embedder.model.language_model.layers.{new_idx}.{'.'.join(tail_part.split('.')[1:])}"
                new_state_dict[new_key] = value.clone()
                mapped_keys.append(f"VLM: {key} -> {new_key}")
            else:
                skipped_keys.append(key)
        else:
            # Copy as is for anything else
            new_state_dict[key] = value.clone()
            
    logging.info(f"Summary: Mapped {len(mapped_keys)} specific layer keys.")
    logging.info(f"Summary: Skipped {len(skipped_keys)} unused layer keys.")
    logging.info(f"Total keys in new state dict: {len(new_state_dict)}")
    
    out_dict = {"module": new_state_dict} if is_module else new_state_dict
    torch.save(out_dict, output_path)
    logging.info(f"Successfully saved customized student weights to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer weights from Teacher to Student using Shallow-pi constraints.")
    parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to mp_rank_00_model_states.pt or pytorch_model.bin")
    parser.add_argument("--output_ckpt", type=str, required=True, help="Path to save the new student weights")
    parser.add_argument("--action_head_map", type=str, default="0:0,2:1,4:2,6:3", help="Map teacher action head layers to student layers. Format: t0:s0,t1:s1")
    parser.add_argument("--vlm_map", type=str, default="0:0,4:1,8:2,13:3", help="Map teacher VLM layers to student layers. Format: t0:s0,t1:s1")
    
    args = parser.parse_args()
    
    action_head_map = {int(k): int(v) for k,v in (pair.split(":") for pair in args.action_head_map.split(","))}
    vlm_map = {int(k): int(v) for k,v in (pair.split(":") for pair in args.vlm_map.split(","))}
    
    logging.info(f"Action Head mapping: {action_head_map}")
    logging.info(f"VLM mapping: {vlm_map}")
    
    transfer_weights(args.teacher_ckpt, args.output_ckpt, action_head_map, vlm_map)
