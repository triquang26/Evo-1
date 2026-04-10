import torch
import torch.nn as nn
from src.evo.training.trainer import Trainer

class SnapFlowTrainer(Trainer):
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def build_models(self):
        import os
        import logging

        model = super().build_models()

        if self.resume_cfg.get("enabled", False):
            resume_path = self.resume_cfg.get("path")
            ckpt_path = os.path.join(resume_path, "mp_rank_00_model_states.pt")
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(checkpoint.get("module", checkpoint), strict=False)
                if self.accelerator.is_main_process:
                    logging.info(f"Loaded Teacher Pretrained Weights from {ckpt_path}")
                
                # Disable accelerator.load_state downstream so we start fresh distillation at Step 0
                self.resume_cfg["enabled"] = False
            else:
                if self.accelerator.is_main_process:
                    logging.warning(f"No PyTorch ckpt found at {ckpt_path}. Check resume path.")
                    
        return model

    def prepare_data(self):
        fm_cfg = self.model_cfg.get("flowmatching", self.model_cfg)
        correct_horizon = fm_cfg.get("action_horizon", fm_cfg.get("horizon", 16))
        
        # Inject correct horizon to root to bypass parent's hardcoded constraint
        self.model_cfg["horizon"] = correct_horizon
        return super().prepare_data()

    def compute_loss(self, models, batch, step):
        model = models
        states = batch["states"].to(dtype=torch.bfloat16)
        actions_gt = batch["actions"].to(dtype=torch.bfloat16)
        action_mask = batch["action_mask"]
        B = states.size(0)
        device = states.device

        fused_tokens_list = []
        grad_ctx = torch.no_grad() if not self.train_cfg.get("finetune_vlm", False) else torch.enable_grad()

        with grad_ctx:
            for prompt, images, im_mask in zip(batch["prompts"], batch["images"], batch["image_masks"]):
                fused = model.get_vl_embeddings(images=images, image_mask=im_mask, prompt=prompt, return_cls_only=False)
                fused_tokens_list.append(fused.to(dtype=torch.bfloat16))

        fused_tokens = torch.cat(fused_tokens_list, dim=0)

        alpha = self.train_cfg.get("alpha", 0.5)
        lambda_weight = self.train_cfg.get("lambda_weight", 0.1)
        clamp_range = self.train_cfg.get("prediction_clamp_range", [-20, 20])

        total_loss = 0.0
        metrics = {}

        action_mask_flat = action_mask.view(B, -1).to(dtype=torch.bfloat16)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # ---------------------------------------------------------
            # 1. Flow Matching Component Loss (L_FM)
            # ---------------------------------------------------------
            # Standard Flow Matching Forward Pass
            # Uses random t and s=None (no target-time embedding)
            pred_velocity_fm, noise_fm = model(
                fused_tokens, state=states, actions_gt=actions_gt, action_mask=action_mask, s=None
            )
            target_velocity_fm = (actions_gt - noise_fm).view(B, -1).detach()
            pred_velocity_fm_mask = pred_velocity_fm * action_mask_flat
            
            loss_fn = nn.MSELoss()
            loss_fm = loss_fn(pred_velocity_fm_mask, target_velocity_fm) * (action_mask_flat.numel() / (action_mask_flat.sum() + 1e-8))
            
            # ---------------------------------------------------------
            # 2. Consistency Short-Cut Component Loss (L_shortcut)
            # ---------------------------------------------------------
            # Student path
            x_0 = torch.randn_like(actions_gt) # t=0 is pure noise in Evo-1
            if action_mask is not None:
                x_0 = x_0 * action_mask.to(dtype=x_0.dtype)

            # Two-Step Euler from Teacher Path (requires stop-gradient)
            with torch.no_grad():
                # Step 1 at t=0
                t_0 = torch.zeros(B, device=device, dtype=torch.bfloat16)
                # s=None means normal behaviour without Target-Time embedding
                v_0, _ = model(
                    fused_tokens, state=states, actions_gt=x_0, noise=x_0, action_mask=action_mask, t=t_0, s=None
                )
                
                # Midpoint (Evo-1 integrates forward, so addition)
                x_05 = x_0 + 0.5 * v_0.view_as(x_0)
                if action_mask is not None:
                    x_05 = x_05 * action_mask.to(dtype=x_05.dtype)

                # Step 2 at t=0.5
                t_05 = torch.full((B,), 0.5, device=device, dtype=torch.bfloat16)
                v_05, _ = model(
                    fused_tokens, state=states, actions_gt=x_05, noise=x_05, action_mask=action_mask, t=t_05, s=None
                )
                
                # Target Short-cut Velocity
                v_target = 0.5 * (v_0 + v_05)
                v_target_flat = v_target.view(B, -1).detach()

            # Student prediction path: 1-step, activate Target-Time Embedding with s=1 (clean), current time t=0 (noise)
            s_1 = torch.ones(B, device=device, dtype=torch.bfloat16)
            pred_velocity_student, _ = model(
                fused_tokens.detach(), state=states, actions_gt=x_0, noise=x_0, action_mask=action_mask, t=t_0, s=s_1
            )

            # Apply Clamping to stop explosion early in training
            pred_velocity_student = torch.clamp(pred_velocity_student, min=clamp_range[0], max=clamp_range[1])
            
            pred_velocity_student_mask = pred_velocity_student * action_mask_flat
            v_target_mask = v_target_flat * action_mask_flat

            loss_shortcut = loss_fn(pred_velocity_student_mask, v_target_mask) * (action_mask_flat.numel() / (action_mask_flat.sum() + 1e-8))

            # Total Loss Combination
            total_loss = (alpha * loss_fm) + ((1 - alpha) * lambda_weight * loss_shortcut)

        metrics = {
            "loss": total_loss.item(),
            "loss_fm": loss_fm.item(),
            "loss_shortcut": loss_shortcut.item()
        }

        check_info = {
            "states": states, 
            "actions_gt": actions_gt, 
            "fused": fused_tokens, 
            "pred_fm": pred_velocity_fm, 
            "pred_student": pred_velocity_student,
            "loss": total_loss
        }

        return total_loss, metrics, check_info

def main():
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train/snapflow_libero.yaml"
    trainer = SnapFlowTrainer(config_path)
    trainer.train()

if __name__ == "__main__":
    main()
