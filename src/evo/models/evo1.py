import torch
import torch.nn as nn
from typing import List, Union, Tuple
from PIL import Image
from pathlib import Path
import yaml

from src.evo.models.components.vision_encoders.internvl3_embedder import InternVL3Embedder
from src.evo.models.components.action_heads.flow_matching import FlowmatchingActionHead
from types import SimpleNamespace

class EVO1(nn.Module):
    def __init__(self, config: Union[str, Path, dict]):
        super().__init__()
        
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.is_absolute():
                # Tự động map về workspace configs nếu truyền path tương đối
                project_root = Path(__file__).resolve().parents[3]
                config_path = project_root / "configs" / config_path
                
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        self._device = self.config.get("device", "cuda")
        self.return_cls_only = self.config.get("return_cls_only", False)
        
        model_cfg = self.config.get("model", self.config)
        vlm_name = model_cfg.get("vision_encoder", "OpenGVLab/InternVL3-1B")
        self.embedder = InternVL3Embedder(model_name=vlm_name, device=self._device)

        action_head_type = model_cfg.get("action_head", "flowmatching").lower()
        
        if action_head_type == "flowmatching":
            fm_cfg = model_cfg.get("flowmatching", model_cfg)
            horizon = fm_cfg.get("action_horizon", fm_cfg.get("horizon", 16))
            per_action_dim = fm_cfg.get("per_action_dim", 7)
            action_dim = horizon * per_action_dim
            
            if action_dim != horizon * per_action_dim:
                raise ValueError(f"action_dim ({action_dim}) ≠ horizon ({horizon}) × per_action_dim ({per_action_dim})")
            
            self.horizon = horizon
            self.per_action_dim = per_action_dim
            
            head_ns = SimpleNamespace(
                embed_dim=fm_cfg.get("embed_dim", 896),    
                hidden_dim=fm_cfg.get("hidden_dim", 1024),
                action_dim=action_dim,
                horizon=horizon,
                per_action_dim=per_action_dim,
                state_dim=fm_cfg.get("state_dim", 7),
                state_hidden_dim=fm_cfg.get("state_hidden_dim", 1024),
                num_heads=fm_cfg.get("num_heads", 8),
                num_layers=fm_cfg.get("num_layers", 8),
                dropout=fm_cfg.get("dropout", 0.0),
                num_inference_timesteps=fm_cfg.get("num_inference_timesteps", 50),
                num_categories=fm_cfg.get("num_categories", 1)
            )
            self.action_head = FlowmatchingActionHead(config=head_ns).to(self._device)
        else:
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")

    def get_vl_embeddings(
        self,
        images: List[Image.Image],
        image_mask: torch.Tensor,  
        prompt: str = "",
        return_cls_only: Union[bool, None] = None
    ) -> torch.Tensor:
        if return_cls_only is None:
            return_cls_only = self.return_cls_only

        if not images:
            raise ValueError("Must provide at least one image (PIL.Image). Got `images=None` or empty list.")
            
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=return_cls_only,
        )

    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input, dtype=torch.float32)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError("Unsupported state input type")

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor.to(self._device)

    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if actions_gt is None:
            return self.action_head.get_action(fused_tokens, state=state, action_mask=action_mask, embodiment_id=embodiment_ids)
        return self.action_head(fused_tokens, state=state, actions_gt=actions_gt, action_mask=action_mask, embodiment_id=embodiment_ids)

    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        import time
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        fused_tokens = self.get_vl_embeddings(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            return_cls_only=return_cls_only
        )
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        state_tensor = self.prepare_state(state_input)  
        action = self.predict_action(fused_tokens, state_tensor, action_mask=action_mask)
        
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        print(f"[Profiling 1] VLM InternVL3: {(t1 - t0) * 1000:.2f} ms | Flow Matching Head: {(t2 - t1) * 1000:.2f} ms")

        return action

    def forward(self, fused_tokens, state=None, actions_gt=None, action_mask=None, embodiment_ids=None):
        return self.predict_action(fused_tokens, state, actions_gt, action_mask, embodiment_ids)

    def _freeze_module(self, module: nn.Module, name: str):
        for p in module.parameters():
            p.requires_grad = False

    def set_finetune_flags(self):
        train_cfg = self.config.get("train", self.config)
        if not train_cfg.get("finetune_vlm", False):
            self._freeze_module(self.embedder, "VLM (InternVL3)")

        if not train_cfg.get("finetune_action_head", False):
            self._freeze_module(self.action_head, "Action Head")
