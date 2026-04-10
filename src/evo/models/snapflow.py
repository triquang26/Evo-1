import torch
import torch.nn as nn
from typing import Union, Tuple

from src.evo.models.evo1 import EVO1
from src.evo.models.components.action_heads.snapflow_action_head import SnapFlowActionHead
from types import SimpleNamespace

class SnapFlowEVO1(EVO1):
    def __init__(self, config):
        import copy
        # Clone config to avoid mutating globally and trick parent class
        safe_config = copy.deepcopy(config)
        model_cfg = safe_config.get("model", safe_config)
        
        action_head_type = model_cfg.get("action_head", "flowmatching").lower()
        if action_head_type == "snapflow":
            model_cfg["action_head"] = "flowmatching"
            
        # Allow default initialization from EVO1 (which now sees "flowmatching")
        super().__init__(safe_config)
        
        # Restore the original check for our SnapFlow specific initialization
        if action_head_type == "snapflow":
            # fm_cfg reading from original config
            orig_model_cfg = config.get("model", config)
            fm_cfg = orig_model_cfg.get("flowmatching", orig_model_cfg)
            horizon = fm_cfg.get("action_horizon", fm_cfg.get("horizon", 16))
            per_action_dim = fm_cfg.get("per_action_dim", 7)
            action_dim = horizon * per_action_dim
            
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
                num_inference_timesteps=fm_cfg.get("num_inference_timesteps", 1), 
                num_categories=fm_cfg.get("num_categories", 1)
            )
            # Override action_head with the SnapFlow one
            self.action_head = SnapFlowActionHead(config=head_ns).to(self._device)

    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
        return_attn_weights: bool = False,
        t: torch.Tensor = None,
        s: torch.Tensor = None,
        noise: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if actions_gt is None:
            return self.action_head.get_action(fused_tokens, state=state, embodiment_id=embodiment_ids, action_mask=action_mask, s=s, t=t)
        return self.action_head(fused_tokens, state=state, actions_gt=actions_gt, action_mask=action_mask, embodiment_id=embodiment_ids, return_attn_weights=return_attn_weights, t=t, s=s, noise=noise)

    def forward(self, fused_tokens, state=None, actions_gt=None, action_mask=None, embodiment_ids=None, return_attn_weights=False, t=None, s=None, noise=None):
        return self.predict_action(fused_tokens, state, actions_gt, action_mask, embodiment_ids, return_attn_weights=return_attn_weights, t=t, s=s, noise=noise)

    def set_finetune_flags(self):
        super().set_finetune_flags()
        # In case we need specific logic for Snapflow. Currently the same as EVO1
