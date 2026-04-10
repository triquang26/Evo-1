import torch
import torch.nn as nn
from src.evo.models.components.action_heads.flow_matching import FlowmatchingActionHead

class SnapFlowActionHead(FlowmatchingActionHead):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        
        # Target-Time Embedding phi_s
        # 2-layer MLP zero-initialized
        self.target_time_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(), 
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Zero initialization to preserve original model behavior at s=0
        # Initialize ONLY the final layer to prevent dead gradients in SiLU
        nn.init.zeros_(self.target_time_mlp[-1].weight)
        nn.init.zeros_(self.target_time_mlp[-1].bias)
                
    def forward(self, fused_tokens: torch.Tensor, state: torch.Tensor = None,
                actions_gt: torch.Tensor = None, embodiment_id: torch.LongTensor = None, 
                state_mask: torch.Tensor = None, action_mask: torch.Tensor = None,
                return_attn_weights: bool = False, t: torch.Tensor = None, s: torch.Tensor = None, noise: torch.Tensor = None):
        
        if actions_gt is None:
            return self.get_action(fused_tokens, state=state, embodiment_id=embodiment_id, action_mask=action_mask, s=s)

        B = fused_tokens.size(0)
        device = fused_tokens.device

        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        context_tokens = fused_tokens 
        if state is not None and self.state_encoder is not None:
            state_emb = self.state_encoder(state, embodiment_id)  
            state_emb = state_emb.unsqueeze(1) 
            context_tokens = torch.cat([context_tokens, state_emb], dim=1) 

        if t is None:
            t = torch.distributions.Beta(2, 2).sample((B,)).clamp(0.02, 0.98).to(device).to(dtype=self.dtype)

        time_index = (t * 1000).long().clamp(0, 999)  
        time_emb = self.time_pos_enc(1000)[:, time_index, :].squeeze(0) 

        # SnapFlow target-time embedding injection
        if s is not None:
            s_index = (s * 1000).long().clamp(0, 999)
            s_emb = self.time_pos_enc(1000)[:, s_index, :].squeeze(0)
            target_time_emb = self.target_time_mlp(s_emb)
            time_emb = time_emb + target_time_emb

        actions_gt_seq = actions_gt  

        if noise is None:
            noise = torch.randn_like(actions_gt) 
            if action_mask is not None:
                action_mask = action_mask.to(dtype=noise.dtype, device=noise.device)
                noise = noise * action_mask

        if self.horizon > 1:
            noise_seq = noise.view(B, self.horizon, self.per_action_dim)
        else:
            noise_seq = noise.unsqueeze(1)

        if self.horizon > 1:
            t_broadcast = t.view(B, 1, 1)
        else:
            t_broadcast = t.view(B, 1)
        action_intermediate_seq = (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_seq  

        if self.horizon > 1 and self.action_encoder is not None:
            action_tokens = self.action_encoder(action_intermediate_seq, embodiment_id)  
        else:
            if not hasattr(self, "single_action_proj"):
                self.single_action_proj = nn.Linear(self.per_action_dim, self.embed_dim).to(device)
            action_tokens = self.single_action_proj(action_intermediate_seq) 

        x = action_tokens  
        attn_weights_list = []
        for block in self.transformer_blocks:
            if return_attn_weights:
                x, attn_weights = block(x, context_tokens, time_emb, return_attn_weights=True)
                attn_weights_list.append(attn_weights)
            else:
                x = block(x, context_tokens, time_emb)

        x = self.norm_out(x)  

        if self.horizon > 1:
            x_flat = x.reshape(B, -1)  
            if not hasattr(self, "seq_pool_proj"):
                self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim).to(device)
            x_pooled = self.seq_pool_proj(x_flat)  
        else:
            x_pooled = x.squeeze(1) 

        pred_velocity = self.mlp_head(x_pooled, embodiment_id) 

        if return_attn_weights:
            return pred_velocity, noise, attn_weights_list
        return pred_velocity, noise

    def get_action(self, fused_tokens: torch.Tensor, state: torch.Tensor = None, embodiment_id: torch.LongTensor = None, action_mask: torch.Tensor = None, s: torch.Tensor = None, t: torch.Tensor = None):
        """
        1-NFE Inference for SnapFlow
        At inference time, bypass the 10-step Euler loop.
        Draw noise x_1 ~ N(0, I). 
        Forward once with s=0, t=1 to predict hat(x_0).
        """
        B = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        context_tokens = fused_tokens
        if state is not None and self.state_encoder is not None:
            state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1) 
            context_tokens = torch.cat([context_tokens, state_emb], dim=1)

        action_dim_total = getattr(self.config, "action_dim", None)
        if action_dim_total is None:
            action_dim_total = self.action_dim
       
        if self.horizon > 1:
            per_action_dim = getattr(self.config, "per_action_dim", action_dim_total // self.horizon)
        else:
            per_action_dim = action_dim_total

        action = torch.randn(B, action_dim_total, device=device)
        
        if self.horizon > 1:
            action_seq = action.view(B, self.horizon, per_action_dim)
        else:
            action_seq = action.view(B, 1, per_action_dim)

        if action_mask is not None:
            if action_mask.dim() == 2:
                action_mask = action_mask.view(B, 1, per_action_dim).repeat(1, self.horizon, 1)
            elif action_mask.dim() == 3:
                pass # Already correct shape
                
            action_mask = action_mask.to(dtype=action_seq.dtype, device=action_seq.device)
            assert action_mask.shape == action_seq.shape, f"action_mask shape {action_mask.shape} != action_seq shape {action_seq.shape}"
            action_seq = action_seq * action_mask
        else:
            raise ValueError("action_mask must be provided for inference with flow matching.")

        # 1-step prediction (1-NFE) target time s=1 (clean in Evo-1), current time t=0 (noise in Evo-1)
        t_val = 0.0 if t is None else t
        s_val = 1.0 if s is None else s
        
        t_tensor = torch.full((B,), t_val, device=device, dtype=self.dtype)
        s_tensor = torch.full((B,), s_val, device=device, dtype=self.dtype)
        
        time_index = (t_tensor * 1000).long().clamp(0, 999)
        time_emb = self.time_pos_enc(1000)[:, time_index, :].squeeze(0)  
        
        s_index = (s_tensor * 1000).long().clamp(0, 999)
        s_emb = self.time_pos_enc(1000)[:, s_index, :].squeeze(0)
        
        # Inject Target-Time Embedding
        target_time_emb = self.target_time_mlp(s_emb)
        time_emb = time_emb + target_time_emb
        
        if self.horizon > 1 and self.action_encoder is not None:
            action_seq = action_seq * action_mask
            action_tokens = self.action_encoder(action_seq, embodiment_id) 
        else:
            if hasattr(self, "single_action_proj"):
                action_tokens = self.single_action_proj(action_seq)  
            else:
                self.single_action_proj = nn.Linear(per_action_dim, self.embed_dim).to(device)
                action_tokens = self.single_action_proj(action_seq)

        x = action_tokens
        for block in self.transformer_blocks:
            x = block(x, context_tokens, time_emb)
        x = self.norm_out(x)

        if self.horizon > 1:
            x_flat = x.reshape(B, -1)
            if hasattr(self, "seq_pool_proj"):
                x_pooled = self.seq_pool_proj(x_flat)
            else:
                self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim).to(device)
                x_pooled = self.seq_pool_proj(x_flat)
        else:
            x_pooled = x.squeeze(1)
     
        # Snapflow 1-step prediction: hat{x_0} = x_1 + F(x_1, s=1, t=0)
        # Because in Evo-1, v = x_clean - x_noise, so x_clean = x_noise + v
        pred_velocity = self.mlp_head(x_pooled, embodiment_id)
        action = action + pred_velocity
        return action
