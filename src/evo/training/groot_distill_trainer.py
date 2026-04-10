import os
import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from src.evo.training.trainer import Trainer

# Dynamically add Isaac-GR00T to python path if not present
isaac_path = str(Path(__file__).resolve().parent.parent.parent.parent / "externals" / "Isaac-GR00T")
if isaac_path not in sys.path:
    sys.path.insert(0, isaac_path)

try:
    from gr00t.policy.gr00t_policy import Gr00tPolicy
except ImportError:
    logging.warning("Failed to import Gr00tPolicy. Ensure Isaac-GR00T submodule is initialized.")

class Gr00tMiniDistillationTrainer(Trainer):
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def build_models(self):
        """Khởi tạo Teacher(GR00T) and Student(GR00T-mini)"""
        teacher_cfg = self.cfg.get('teacher_cfg', {})
        teacher_repo = teacher_cfg.get('hf_repo_id', "liorbenhorin-nv/groot-libero_10-64_40000")
        embodiment_tag = self.cfg['model'].get('embodiment_tag', 'LIBERO_PANDA')
        
        logging.info(f"Loading Teacher GR00T from {teacher_repo}")
        self.teacher_policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=teacher_repo,
            device="cpu",
            strict=False
        )
        self.teacher_model = self.teacher_policy.model.eval()
        for param in self.teacher_model.parameters(): 
            param.requires_grad = False

        logging.info("Initializing GR00T-Mini Student from teacher config...")
        teacher_config = self.teacher_model.config
        mini_config = AutoConfig.from_pretrained(teacher_repo)
        
        target_hidden_size = self.cfg['model'].get('hidden_size', 1024)
        target_vlm_layers = len(self.cfg['model'].get('vlm_layer_indices', []))
        target_ah_layers = len(self.cfg['model'].get('action_head_layer_indices', []))

        if hasattr(mini_config, "hidden_size"):
            mini_config.hidden_size = target_hidden_size
            
        if target_vlm_layers > 0:
            mini_config.select_layer = target_vlm_layers
            
        if target_ah_layers > 0 and hasattr(mini_config, "diffusion_model_cfg"):
            mini_config.diffusion_model_cfg["num_layers"] = target_ah_layers
            
        # Bắt đầu đọc file weight
        student_ckpt = self.cfg.get("student_checkpoint_path", "")
        if student_ckpt and os.path.exists(student_ckpt):
            # Because transfer_weights generated a standard HF model, we can load it directly!
            self.student_model = AutoModel.from_pretrained(student_ckpt, trust_remote_code=True)
            logging.info(f"Loaded tailored HF student from {student_ckpt}")
        else:
            self.student_model = AutoModel.from_config(mini_config)
            logging.warning("No student_checkpoint_path found or path does not exist. Initializing with random weights.")
            
        self.student_model.train()
        
        return self.student_model, self.teacher_model

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
        
        # Map Evo-1 dataloader output to GR00T / LeRobot dictionary structure
        gr00t_batch = {
            "observation.state": batch["states"],
            "action": batch["actions"]
        }
        
        images = batch["images"]
        num_views = images.shape[1]
        view_names = self.cfg['model'].get('view_names', ["agentview_image", "robot0_eye_in_hand_image"])
        
        for i in range(min(num_views, len(view_names))):
            gr00t_batch[f"observation.images.{view_names[i]}"] = images[:, i, ...]
            
        dtype = torch.bfloat16
        for k, v in gr00t_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                gr00t_batch[k] = v.to(dtype=dtype, device=self.accelerator.device)
                
        with torch.no_grad():
            teacher_output = teacher_model.forward(gr00t_batch)
            
        student_output = student_model.forward(gr00t_batch)
        
        loss_fn = torch.nn.L1Loss()
        lambda_task = self.train_cfg.get("lambda_task", 1.0)
        
        loss_task = 0.0
        
        if isinstance(student_output, dict) and "action_pred" in student_output:
            val_stu = student_output["action_pred"]
            val_tea = teacher_output["action_pred"]
            loss_task = loss_fn(val_stu, val_tea)
        elif isinstance(student_output, tuple):
            loss_task = loss_fn(student_output[0], teacher_output[0])
        else:
            loss_task = loss_fn(student_output, teacher_output)
            
        loss_task = loss_task * lambda_task

        metrics = {
            "loss": loss_task.item(),
            "l_task": loss_task.item(),
        }
        
        pred_val = None
        if isinstance(student_output, dict) and "action_pred" in student_output:
            pred_val = student_output["action_pred"]
        elif isinstance(student_output, tuple):
            pred_val = student_output[0]
        else:
            pred_val = student_output

        check_info = {
            "pred": pred_val,
            "loss": loss_task
        }
        
        return loss_task, metrics, check_info

def main():
    import sys
    import os
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    os.chdir(project_root)
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train/distill_groot.yaml"
    trainer = Gr00tMiniDistillationTrainer(config_path)
    trainer.train()

if __name__ == "__main__":
    main()

