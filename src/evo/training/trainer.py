import sys
import os
import math
import time
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
import yaml
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from accelerate import Accelerator, DistributedType
import wandb
import swanlab

from src.evo.models.evo1 import EVO1

class Trainer:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._load_config()
        self.accelerator = Accelerator()
        self._setup_logging()
        self._init_trackers()

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Merge sub-configs for easier flat access in specific spots if needed
        self.dataset_cfg = self.cfg.get('dataset', {})
        self.train_cfg = self.cfg.get('train', {})
        self.log_cfg = self.cfg.get('logging', {})
        self.model_cfg = self.cfg.get('model', {})
        self.resume_cfg = self.cfg.get('resume', {})

    def _setup_logging(self):
        save_dir = self.log_cfg.get("save_dir", "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(save_dir, f"train_{timestamp}.log")

        if self.accelerator.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.FileHandler(self.log_path), logging.StreamHandler()]
            )
            logging.info(f"Initialized training logging at: {self.log_path}")

    def _init_trackers(self):
        if self.accelerator.is_main_process:
            if self.cfg.get("disable_wandb", False):
                os.environ["WANDB_MODE"] = "disabled"
            
            wandb.init(
                project=self.cfg.get("wandb_project", "default_run"),
                name=self.cfg.get("run_name", "default_run"),
                config=self.cfg,
                dir=self.log_cfg.get("save_dir", "checkpoints"),
                mode="offline"
            )
            wandb.define_metric("*", step_metric="step")
            
            swanlab.init(
                project=self.cfg.get("wandb_project", "default_run"),
                name=self.cfg.get("run_name", "default_run"),
                config=self.cfg
            )

    @staticmethod
    def custom_collate_fn(batch):
        return {
            "prompts": [item["prompt"] for item in batch],
            "images": [item["images"] for item in batch],
            "states": torch.stack([item["state"] for item in batch], dim=0),
            "actions": torch.stack([item["action"] for item in batch], dim=0),
            "action_mask": torch.stack([item["action_mask"] for item in batch], dim=0),
            "state_mask": torch.stack([item["state_mask"] for item in batch], dim=0),
            "image_masks": torch.stack([item["image_mask"] for item in batch], dim=0),
            "embodiment_ids": torch.stack([item["embodiment_id"] for item in batch], dim=0)
        }

    def prepare_data(self):
        # Allow dynamic resolution of old vs new dataset locations
        try:
            from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset 
        except ImportError:
            from src.evo.data.lerobot_dataset import LeRobotDataset
        
        with open(self.dataset_cfg.get("config_path"), 'r') as f:
            ds_cfg = yaml.safe_load(f)

        dataset = LeRobotDataset(
            config=ds_cfg,
            image_size=self.dataset_cfg.get("image_size", 448),
            max_samples_per_file=self.dataset_cfg.get("max_samples_per_file"),
            action_horizon=self.model_cfg.get("horizon", 16),
            binarize_gripper=self.dataset_cfg.get("binarize_gripper", False),
            use_augmentation=self.dataset_cfg.get("use_augmentation", False)
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.dataset_cfg.get("batch_size", 16),
            shuffle=True,
            num_workers=self.dataset_cfg.get("num_workers", 8),
            pin_memory=True,
            persistent_workers=(self.dataset_cfg.get("num_workers", 0) > 0),
            drop_last=True,
            collate_fn=self.custom_collate_fn
        )
        
        if self.accelerator.is_main_process:
            logging.info(f"Loaded {len(dataset)} samples. Batch size: {dataloader.batch_size}")
        
        return dataset, dataloader

    def _get_lr_lambda(self, warmup_steps, total_steps, resume_step=0):
        def lr_lambda(current_step):
            current_step += resume_step  
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr_lambda

    def check_inf(self, step, **tensors):
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                logging.info(f"[Step {step}] Non-finite detected in {name}")
                return False
        return True

    def train(self):
        dataset, dataloader = self.prepare_data()
        
        model = EVO1(self.cfg)
        model.train()
        model.set_finetune_flags()

        lr = self.train_cfg.get("lr", 1e-4)
        wd = self.train_cfg.get("weight_decay", 1e-5)
        
        # Build param groups
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if n.endswith("bias") or ".bias" in n or p.dim() == 1 or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        
        optimizer = AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)

        model, optimizer, dataloader = self.accelerator.prepare(model, optimizer, dataloader)

        max_steps = self.train_cfg.get("max_steps", 60000)
        warmup_steps = self.train_cfg.get("warmup_steps", 3000)
        loss_fn = nn.MSELoss()
        
        step, best_loss = 0, float("inf")
        save_dir = self.log_cfg.get("save_dir", "checkpoints")
        
        if self.resume_cfg.get("enabled", False):
            resume_path = self.resume_cfg.get("path")
            self.accelerator.load_state(resume_path)
            meta_path = os.path.join(resume_path, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                step = meta.get("step", 0)
                best_loss = meta.get("best_loss", float("inf"))
            if self.accelerator.is_main_process:
                logging.info(f"Resumed from {resume_path} at step {step}")

        scheduler = LambdaLR(optimizer, self._get_lr_lambda(warmup_steps, max_steps, resume_step=step))

        while step < max_steps:
            for batch in tqdm(dataloader, desc="Training", disable=not self.accelerator.is_main_process):
                if step >= max_steps: break

                states = batch["states"].to(dtype=torch.bfloat16)
                actions_gt = batch["actions"].to(dtype=torch.bfloat16)
                action_mask = batch["action_mask"]
                
                fused_tokens_list = []
                grad_ctx = torch.no_grad() if not self.train_cfg.get("finetune_vlm", False) else torch.enable_grad()
                
                with grad_ctx:
                    for prompt, images, im_mask in zip(batch["prompts"], batch["images"], batch["image_masks"]):
                        fused = model.get_vl_embeddings(images=images, image_mask=im_mask, prompt=prompt, return_cls_only=False)
                        fused_tokens_list.append(fused.to(dtype=torch.bfloat16))

                fused_tokens = torch.cat(fused_tokens_list, dim=0)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_velocity, noise = model(fused_tokens, state=states, actions_gt=actions_gt, action_mask=action_mask)
                    
                target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
                
                action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
                pred_velocity_mask = pred_velocity * action_mask
                loss = loss_fn(pred_velocity_mask, target_velocity) * (action_mask.numel() / (action_mask.sum() + 1e-8))

                if not self.check_inf(step, states=states, actions_gt=actions_gt, fused=fused_tokens, pred=pred_velocity, loss=loss):
                    continue

                optimizer.zero_grad(set_to_none=True)
                self.accelerator.backward(loss)
                
                if hasattr(self.accelerator, "clip_grad_norm_"):
                    self.accelerator.clip_grad_norm_(model.parameters(), self.train_cfg.get("grad_clip_norm", 1.0))
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_cfg.get("grad_clip_norm", 1.0))

                optimizer.step()
                scheduler.step()

                if step % self.log_cfg.get("log_interval", 100) == 0 and self.accelerator.is_main_process:
                    curr_ep = step / len(dataloader)
                    logging.info(f"[Step {step}] Loss: {loss.item():.4f}")
                    metrics = {"step": step, "loss": loss.item(), "epoch": curr_ep, "lr": scheduler.get_last_lr()[0]}
                    wandb.log(metrics)
                    swanlab.log(metrics)

                # Checkpoint saving block
                if self.accelerator.is_main_process:
                    is_best = loss.item() < best_loss
                    if is_best: best_loss = loss.item()
                else:
                    is_best = False

                is_best_tensor = torch.tensor(int(is_best), device=self.accelerator.device)
                if self.accelerator.distributed_type != DistributedType.NO:
                    torch.distributed.broadcast(is_best_tensor, src=0)

                should_save_best = (is_best_tensor.item() == 1 and step > 1000)
                should_save_periodic = (step % self.log_cfg.get("ckpt_interval", 1000) == 0 and step > 0)
                
                if should_save_best:
                    self._save_checkpoint("best", loss.item(), getattr(dataset, "arm2stats_dict", None))
                if should_save_periodic:
                    self._save_checkpoint(step, loss.item(), getattr(dataset, "arm2stats_dict", None))

                step += 1

        self._save_checkpoint("final", loss.item(), getattr(dataset, "arm2stats_dict", None))
        logging.info("Training complete.")

    def _save_checkpoint(self, tag, loss_val, norm_stats=None):
        out_dir = os.path.join(self.log_cfg.get("save_dir", "checkpoints"), f"step_{tag}")
        if self.accelerator.is_main_process and os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            
        self.accelerator.wait_for_everyone()
        os.makedirs(out_dir, exist_ok=True)
        self.accelerator.save_state(out_dir)
        
        if self.accelerator.is_main_process:
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump({"step": tag, "best_loss": loss_val}, f, indent=2)
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(self.cfg, f, indent=2)
            if norm_stats:
                with open(os.path.join(out_dir, "norm_stats.json"), "w") as f:
                    json.dump(norm_stats, f, indent=2)
            logging.info(f"Saved checkpoint: {out_dir}")

def main():
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train/finetune_full.yaml"
    trainer = Trainer(config_path)
    trainer.train()

if __name__ == "__main__":
    main()
