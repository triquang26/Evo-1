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

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["MUJOCO_GL"] = "osmes"

from src.evo.models.builder import build_model
from src.evo.data.lerobot_dataset import LeRobotDataset

# Tắt Flash Attention để mô hình trả về Attention bằng Fallback (Sdpa/Eager)
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
            raise ValueError(f"Không đủ lượng token hình ảnh để visualize. Có: {total_img_tokens}")

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
            print(f"-> Đã lưu ảnh Visualize Attention tại: {save_path}")
            
        return overlaid_img

# ==============================================================================
# LOGIC THỰC THI (HỖ TRỢ DATASET LEROBOT VÀ EVO1_STUDENT)
# ==============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. CONFIG VÀ KHỞI TẠO MODEL TỪ CHECKPOINT ---
    checkpoint_path = "/mnt/data/sftp/data/quangpt3/Evo-1/checkpoint_distill_stage2_40000_v02-3/step_22000"
    pt_path = os.path.join(checkpoint_path, "mp_rank_00_model_states.pt")
    
    # Để nạp đúng tensor dimensions từ checkpoint, bắt buộc phải dùng config lúc train (không dùng config inference chung)
    train_cfg_path = "configs/train/distill.yaml"
    with open(train_cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)
        
    train_cfg["device"] = device
    
    print("Building model...")
    model = build_model(train_cfg)
    model.eval()
    
    print(f"Loading checkpoint từ {pt_path} ...")
    state_dict = torch.load(pt_path, map_location="cpu")
    if "module" in state_dict:
        state_dict = state_dict["module"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # --- Không cần gắn Hook vì ta đã đổi Model Core trả về tự nhiên ---
    # model.embedder.last_attentions sẽ lưu tự động sau khi inference


    # --- 3. ĐỌC DỮ LIỆU TỪ DATASET LEROBOT (CHUẨN XÁC THEO TRAIN LOOP) ---
    # Lấy action_horizon từ cấu hình lúc train
    action_horizon = train_cfg["model"]["flowmatching"]["action_horizon"]
    
    # Ở distill_trainer.py, cấu hình data config được đọc từ đường dẫn bên trong dataset config
    dataset_cfg_path = train_cfg["dataset"].get("config_path", "src/evo/data/config.yaml")
    with open(dataset_cfg_path, "r") as f:
        dataset_inner_cfg = yaml.safe_load(f)
    
    dataset = LeRobotDataset(
        config=dataset_inner_cfg,
        image_size=448,
        action_horizon=action_horizon
    )
    
    # Lấy sample đầu tiên
    sample = dataset[0]
    
    # Giữ nguyên toàn bộ ds ảnh (kể cả dummy pad) để len(images) khớp len(image_mask)
    images = [img for img in sample["images"]]
    
    prompt = sample["prompt"]
    image_masks = sample["image_mask"].unsqueeze(0) # (1, max_views) 
    state = sample["state"].unsqueeze(0)
    
    print(f"Prompt của dữ liệu: {prompt}")

    # --- 4. TÌM VỊ TRÍ CÁC TOKEN HÌNH ẢNH (IMG_CONTEXT) ---
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

    # --- 5. CHẠY FORWARD PASS ĐỂ KÍCH HOẠT HOOK ---
    print("Running forward pass...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Model inference:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(display_image_tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(display_image_tensor.device)
            
            # Khử chuẩn hoá về pixel [0, 255] dùng cho việc trực quan OpenCV
            img_unnorm = display_image_tensor * std + mean
            img_pil = Image.fromarray((img_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            
            # Forward toàn bộ ds ảnh (kể cả dummies) với image_mask tương ứng
            fused_tokens = model.get_vl_embeddings(
                images=images,
                image_mask=sample["image_mask"],
                prompt=prompt,
                return_cls_only=False
            )
            # Không cần chạy hết action head, get_vl_embeddings đã thực thi LLM xong !
            # Hook đã được trigger !
            
            # Lấy thông tin padding và tạo model_inputs ảo để mapping token index
            _, num_tiles_list = model.embedder._preprocess_images(images)
            text_with_img = model.embedder._build_multimodal_prompt(num_tiles_list, prompt)
            model_inputs = model.embedder.tokenizer(
                text_with_img, return_tensors="pt", max_length=1024, padding="max_length", truncation=True
            )
            base_attn_mask = model_inputs["attention_mask"][0]
            
            # Khớp logic image_mask = 0 của InternVL3Embedder
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

    # --- 6. TRÍCH XUẤT VÀ VẼ OVERLAY TỪ NATIVE MODEL HUGGINGFACE ---
    if not hasattr(model.embedder, "last_attentions") or model.embedder.last_attentions is None:
        raise RuntimeError("Model không xuất ra được attention. Vui lòng kiểm tra lại setup Model (FlashAttention).")
        
    attn_weights = model.embedder.last_attentions[-1].detach().cpu() # Lấy attention ở layer cuối cùng

    
    # Tìm index thật của chuỗi dựa vào base_attn_mask
    # Chú ý: Ở giữa chuỗi có thể có số 0 (do ảnh bị mask), vị trí query cuối cùng là số 1 cuối cùng.
    last_query_idx = base_attn_mask.nonzero()[-1].item()
    
    # Tại query token cuối cùng, lấy attention tới tất cả các token khác: shape (1, num_heads, kv_len)
    final_token_attn = attn_weights[0, :, last_query_idx, :]
    
    img_indices = img_token_locations

    if len(img_indices) == 0:
         print("Cảnh báo: Không tìm thấy token IMG_CONTEXT trong input. Lỗi preprocess?")
         
    print(f"Tổng số IMG_CONTEXT tokens: {len(img_indices)}")
    
    # Do có nhều ảnh (bao gồm dummy), ta cần trích xuất các token tương ứng với `valid_cam_index`
    tokens_per_image = model.embedder.model.num_image_token
    # Ví dụ: num_tiles_list = [1, 1, 1] nghĩa là mỗi hình 1 tile.
    start_token_idx = sum(num_tiles_list[:valid_cam_index]) * tokens_per_image
    end_token_idx = start_token_idx + num_tiles_list[valid_cam_index] * tokens_per_image
    
    target_img_indices = img_indices[start_token_idx:end_token_idx]

    processor = AttentionPostProcessor(tokens_per_tile=model.embedder.model.num_image_token)
    heatmap_2d = processor.tokens_to_2d_heatmap(final_token_attn, target_img_indices)
    
    output_path = "attention_visualization.jpg"
    OverlayVisualizer.overlay_on_image(img_pil, heatmap_2d, alpha=0.5, save_path=output_path)
    
if __name__ == "__main__":
    main()
