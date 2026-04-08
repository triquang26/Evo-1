import os
import torch
import torch.nn as nn
from torch.optim import AdamW
import logging
from src.evo.training.trainer import Trainer
from src.evo.models.builder import build_model

class DistillationTrainer(Trainer):
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def build_models(self):
        # 1. Build Student Model
        student_model = build_model(self.cfg)
        student_model.train()
        student_model.set_finetune_flags()
        
        student_ckpt = self.cfg.get("student_checkpoint_path", None)
        if student_ckpt and os.path.exists(student_ckpt):
            try:
                state_dict = torch.load(student_ckpt, map_location="cpu")
                if "module" in state_dict:
                    state_dict = state_dict["module"]
                student_model.load_state_dict(state_dict, strict=False)
                logging.info(f"Loaded Student from {student_ckpt}")
            except Exception as e:
                logging.warning(f"Failed to load Student checkpoint from {student_ckpt}: {e}")

        # 2. Build Teacher Model
        teacher_cfg_dict = self.cfg.get('teacher_cfg', self.cfg)
        teacher_model = build_model(teacher_cfg_dict)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        teacher_ckpt = self.cfg.get("teacher_checkpoint_path", None)
        if teacher_ckpt:
            try:
                ckpt_path = os.path.join(teacher_ckpt, "mp_rank_00_model_states.pt")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(teacher_ckpt, "pytorch_model.bin")
                state_dict = torch.load(ckpt_path, map_location="cpu")
                if "module" in state_dict:
                    state_dict = state_dict["module"]
                teacher_model.load_state_dict(state_dict, strict=False)
                logging.info(f"Loaded Teacher from {ckpt_path}")
            except Exception as e:
                logging.warning(f"Failed to load Teacher checkpoint from {teacher_ckpt}: {e}")

        return student_model, teacher_model

    def setup_optimizer(self, models):
        student_model, _ = models
        return super().setup_optimizer(student_model)

    def prepare_accelerator(self, models, optimizer, dataloader):
        student_model, teacher_model = models
        teacher_model = teacher_model.to(self.accelerator.device)
        student_model, optimizer, dataloader = self.accelerator.prepare(
            student_model, optimizer, dataloader
        )
        return (student_model, teacher_model), optimizer, dataloader
        
    def get_model_to_step(self, models):
        student_model, _ = models
        return student_model

    def compute_loss(self, models, batch, step):
        student_model, teacher_model = models
        states = batch["states"].to(dtype=torch.bfloat16)
        actions_gt = batch["actions"].to(dtype=torch.bfloat16)
        action_mask = batch["action_mask"]
        
        lambda_task = self.train_cfg.get("lambda_task", 0.1)
        lambda_kd_vel = self.train_cfg.get("lambda_kd_vel", 1.0)
        lambda_kd_attn = self.train_cfg.get("lambda_kd_attn", 1.0)
        
        student_fused_tokens_list = []
        teacher_fused_tokens_list = []
        
        grad_ctx = torch.no_grad() if not self.train_cfg.get("finetune_vlm", False) else torch.enable_grad()
        with grad_ctx:
            for prompt, images, im_mask in zip(batch["prompts"], batch["images"], batch["image_masks"]):
                stu_fused = student_model.get_projected_vl_embeddings(images=images, image_mask=im_mask, prompt=prompt, return_cls_only=False)
                student_fused_tokens_list.append(stu_fused.to(dtype=torch.bfloat16))
                
        with torch.no_grad():
            for prompt, images, im_mask in zip(batch["prompts"], batch["images"], batch["image_masks"]):
                tea_fused = teacher_model.get_vl_embeddings(images=images, image_mask=im_mask, prompt=prompt, return_cls_only=False)
                teacher_fused_tokens_list.append(tea_fused.to(dtype=torch.bfloat16))

        student_fused_tokens = torch.cat(student_fused_tokens_list, dim=0)
        teacher_fused_tokens = torch.cat(teacher_fused_tokens_list, dim=0)

        B = student_fused_tokens.size(0)
        device = student_fused_tokens.device
        dtype = torch.bfloat16
        shared_t = torch.distributions.Beta(2, 2).sample((B,)).clamp(0.02, 0.98).to(device).to(dtype=dtype)
        
        shared_noise = torch.rand_like(actions_gt) * 2 - 1
        if action_mask is not None:
            action_mask_bfloat = action_mask.to(dtype=shared_noise.dtype, device=shared_noise.device)
            shared_noise = shared_noise * action_mask_bfloat

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_velocity_stu, noise_stu, attn_stu = student_model(
                student_fused_tokens, state=states, 
                actions_gt=actions_gt, action_mask=action_mask, 
                return_attn_weights=True,
                t=shared_t, noise=shared_noise
            )
            
            with torch.no_grad():
                pred_velocity_tea, _, attn_tea = teacher_model(
                    teacher_fused_tokens, state=states, 
                    actions_gt=actions_gt, action_mask=action_mask, 
                    return_attn_weights=True,
                    t=shared_t, noise=shared_noise
                )
            
            student_attn_idx = self.train_cfg.get("kd_attn_student_layer", 1) 
            teacher_attn_idx = self.train_cfg.get("kd_attn_teacher_layer", 3)
            
            attn_s = attn_stu[student_attn_idx]
            attn_t = attn_tea[teacher_attn_idx]
            
            loss_kd_attn = torch.nn.functional.kl_div(
                torch.log(attn_s + 1e-10), 
                attn_t + 1e-10, 
                reduction="batchmean"
            )
            loss_kd_attn = loss_kd_attn / max(1, attn_s.size(-2))
            
            target_velocity = (actions_gt - noise_stu).view(actions_gt.shape[0], -1)
            action_mask_flat = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity_stu.dtype)
            loss_fn = nn.MSELoss()
            
            loss_task = loss_fn(pred_velocity_stu * action_mask_flat, target_velocity) * (action_mask_flat.numel() / (action_mask_flat.sum() + 1e-8))
            loss_kd_vel = loss_fn(pred_velocity_stu * action_mask_flat, pred_velocity_tea * action_mask_flat) * (action_mask_flat.numel() / (action_mask_flat.sum() + 1e-8))
            
            loss = lambda_task * loss_task + lambda_kd_attn * loss_kd_attn + lambda_kd_vel * loss_kd_vel

        metrics = {
            "loss": loss.item(),
            "l_task": loss_task.item(),
            "l_kd_attn": loss_kd_attn.item(),
            "l_kd_vel": loss_kd_vel.item()
        }
        check_info = {
            "states": states,
            "actions_gt": actions_gt,
            "f_stu": student_fused_tokens,
            "pred": pred_velocity_stu,
            "loss": loss
        }
        return loss, metrics, check_info

def main():
    import sys
    import os
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    os.chdir(project_root)
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train/distill.yaml"
    trainer = DistillationTrainer(config_path)
    trainer.train()

if __name__ == "__main__":
    main()
