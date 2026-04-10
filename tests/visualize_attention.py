import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import yaml
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MUJOCO_GL"] = "osmes"

from src.evo.models.builder import build_model
from src.evo.data.lerobot_dataset import LeRobotDataset

# Disable Flash Attention to emit Attention weights via Fallback (Sdpa/Eager)
import src.evo.models.components.vision_encoders.internvl3_embedder as ivl3_embedder
ivl3_embedder.DEBUG = True

class AttentionPostProcessor:
    def __init__(self, tokens_per_tile=256):
        self.grid_size = int(np.sqrt(tokens_per_tile))

    def tokens_to_2d_heatmap(self, target_token_attn: torch.Tensor, img_token_indices: torch.Tensor) -> np.ndarray:
        if target_token_attn.dim() == 2:
            target_token_attn = target_token_attn.mean(dim=0)
            
        img_attn = target_token_attn[img_token_indices]
        total_img_tokens = img_attn.shape[0]
        num_tiles = total_img_tokens // (self.grid_size ** 2)
        
        if num_tiles == 0:
            raise ValueError(f"Not enough image tokens to visualize. Found: {total_img_tokens}")

        img_attn_reshaped = img_attn[:self.grid_size**2].view(self.grid_size, self.grid_size).float().numpy()
        heatmap_norm = img_attn_reshaped / (np.max(img_attn_reshaped) + 1e-8)
        return heatmap_norm

class OverlayVisualizer:
    @staticmethod
    def overlay_on_image(original_image: Image.Image, heatmap_2d: np.ndarray, alpha=0.5, save_path=None):
        img_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap_2d, (img_cv.shape[1], img_cv.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlaid_img = cv2.addWeighted(heatmap_colored, alpha, img_cv, 1 - alpha, 0)
        
        if save_path:
            cv2.imwrite(save_path, overlaid_img)
            print(f"-> Saved Visualization to: {save_path}")
            
        return overlaid_img

def extract_attention(model, images, prompt, sample):
    """
    Runs forward pass, extracts attention and returns heatmap.
    Returns: heatmap_2d (np.ndarray)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = model.embedder.tokenizer
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # Lấy index thực tế của ảnh (view chính) để Extract:
    valid_cam_index = 0
    for i, is_valid in enumerate(sample["image_mask"]):
        if is_valid:
            valid_cam_index = i
            break
            
    display_image_tensor = images[valid_cam_index]

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Forward model để trigger attention hook
            fused_tokens = model.get_vl_embeddings(
                images=images,
                image_mask=sample["image_mask"],
                prompt=prompt,
                return_cls_only=False
            )
            
            # Lấy thông tin padding và mapping vị trí tokens
            _, num_tiles_list = model.embedder._preprocess_images(images)
            text_with_img = model.embedder._build_multimodal_prompt(num_tiles_list, prompt)
            model_inputs = model.embedder.tokenizer(
                text_with_img, return_tensors="pt", max_length=1024, padding="max_length", truncation=True
            )
            base_attn_mask = model_inputs["attention_mask"][0]
            
            img_token_locations = torch.where(model_inputs["input_ids"][0] == img_context_token_id)[0]
            tokens_per_tile = model.embedder.model.num_image_token
            current_token_idx = 0
            for i in range(len(sample["image_mask"])):
                num_tokens_for_this_image = num_tiles_list[i] * tokens_per_tile
                if not sample["image_mask"][i]:
                    start_idx = img_token_locations[current_token_idx]
                    end_idx = start_idx + num_tokens_for_this_image
                    base_attn_mask[start_idx:end_idx] = 0
                current_token_idx += num_tokens_for_this_image

    if not hasattr(model.embedder, "last_attentions") or model.embedder.last_attentions is None:
        raise RuntimeError("Model did not output attention. Check if fallback to eager/sdpa is working.")
        
    attn_weights = model.embedder.last_attentions[-1].detach().cpu()
    
    last_query_idx = base_attn_mask.nonzero()[-1].item()
    final_token_attn = attn_weights[0, :, last_query_idx, :]
    
    img_indices = img_token_locations

    tokens_per_image = model.embedder.model.num_image_token
    start_token_idx = sum(num_tiles_list[:valid_cam_index]) * tokens_per_image
    end_token_idx = start_token_idx + num_tiles_list[valid_cam_index] * tokens_per_image
    
    target_img_indices = img_indices[start_token_idx:end_token_idx]

    processor = AttentionPostProcessor(tokens_per_tile=tokens_per_image)
    heatmap_2d = processor.tokens_to_2d_heatmap(final_token_attn, target_img_indices)
    
    return heatmap_2d

def get_display_image(images, sample):
    valid_cam_index = 0
    for i, is_valid in enumerate(sample["image_mask"]):
        if is_valid:
            valid_cam_index = i
            break
            
    display_image_tensor = images[valid_cam_index]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to("cpu")
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to("cpu")
    
    img_unnorm = display_image_tensor.cpu() * std + mean
    img_pil = Image.fromarray((img_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    return img_pil

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Configurations
    train_cfg_path = "configs/train/distill.yaml"
    with open(train_cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)
        
    train_cfg["device"] = device

    print("Building Student Model...")
    student_model = build_model(train_cfg)
    student_model.eval()
    
    # Checkpoint configuration for Student
    student_pt_path = "/mnt/data/sftp/data/quangpt3/Evo-1/checkpoint_distill_stage2_40000_v02-3/step_30000/mp_rank_00_model_states.pt"
    if not os.path.exists(student_pt_path):
        student_pt_path = train_cfg.get("student_checkpoint_path")
    
    print(f"Loading Student checkpoint from {student_pt_path} ...")
    if student_pt_path and os.path.exists(student_pt_path):
        state_dict = torch.load(student_pt_path, map_location="cpu")
        if "module" in state_dict:
            state_dict = state_dict["module"]
        student_model.load_state_dict(state_dict, strict=False)
    student_model.to(device)

    print("Building Teacher Model...")
    teacher_cfg_dict = train_cfg.get('teacher_cfg', train_cfg)
    teacher_cfg_dict["device"] = device
    teacher_model = build_model(teacher_cfg_dict)
    teacher_model.eval()

    # Checkpoint for Teacher
    teacher_ckpt_dir = "/mnt/data/sftp/data/quangpt3/Evo-1/evaluations/LIBERO/libero_cpkt"
    teacher_pt_path = os.path.join(teacher_ckpt_dir, "mp_rank_00_model_states.pt")
    
    print(f"Loading Teacher checkpoint from {teacher_pt_path} ...")
    if os.path.exists(teacher_pt_path):
        state_dict = torch.load(teacher_pt_path, map_location="cpu")
        if "module" in state_dict:
            state_dict = state_dict["module"]
        teacher_model.load_state_dict(state_dict, strict=False)
    teacher_model.to(device)

    # 3. Dataloader from LeRobotDataset
    action_horizon = train_cfg["model"]["flowmatching"]["action_horizon"]
    dataset_cfg_path = train_cfg["dataset"].get("config_path", "src/evo/data/config.yaml")
    
    with open(dataset_cfg_path, "r") as f:
        dataset_inner_cfg = yaml.safe_load(f)
    
    dataset = LeRobotDataset(
        config=dataset_inner_cfg,
        image_size=448,
        action_horizon=action_horizon
    )
    
    # Create Output Directories
    student_out_dir = "/mnt/data/sftp/data/quangpt3/Evo-1/outputs/attention/children_attention"
    teacher_out_dir = "/mnt/data/sftp/data/quangpt3/Evo-1/outputs/attention/teacher_attention"
    
    os.makedirs(student_out_dir, exist_ok=True)
    os.makedirs(teacher_out_dir, exist_ok=True)

    print("Iterating over 10 dataset steps to visualize attention...")
    
    for step_idx in range(10):
        print(f"--- Step {step_idx+1}/10 ---")
        sample = dataset[step_idx]
        
        images = [img for img in sample["images"]]
        prompt = sample["prompt"]
        
        print(f"Prompt: {prompt}")
        
        # Move inputs to device (images)
        images = [img.to(device) for img in images]
        
        # Get base display Image
        img_pil = get_display_image(images, sample)
        
        # Extract for Teacher
        try:
            heatmap_teacher = extract_attention(teacher_model, images, prompt, sample)
            teacher_save_path = os.path.join(teacher_out_dir, f"step_{step_idx:02d}.jpg")
            OverlayVisualizer.overlay_on_image(img_pil, heatmap_teacher, alpha=0.5, save_path=teacher_save_path)
        except Exception as e:
            print(f"Failed Teacher extraction at step {step_idx}: {e}")

        # Extract for Student
        try:
            heatmap_student = extract_attention(student_model, images, prompt, sample)
            student_save_path = os.path.join(student_out_dir, f"step_{step_idx:02d}.jpg")
            OverlayVisualizer.overlay_on_image(img_pil, heatmap_student, alpha=0.5, save_path=student_save_path)
        except Exception as e:
            print(f"Failed Student extraction at step {step_idx}: {e}")

if __name__ == "__main__":
    main()
