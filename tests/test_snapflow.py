import torch
import torch.nn as nn
from src.evo.models.components.action_heads.snapflow_action_head import SnapFlowActionHead

class Config:
    embed_dim = 16
    hidden_dim = 32
    action_dim = 7
    horizon = 1
    per_action_dim = 7
    num_heads = 1
    num_layers = 1
    dropout = 0.0
    num_inference_timesteps = 1
    num_categories = 1

head = SnapFlowActionHead(config=Config()).to("cpu")
head.train()
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)

print("Step 1")
dummy_s_emb = torch.randn(2, 16)
out = head.target_time_mlp(dummy_s_emb)
loss = out.sum()
loss.backward()

grad_w2 = head.target_time_mlp[-1].weight.grad.abs().mean().item()
grad_w1 = head.target_time_mlp[0].weight.grad.abs().mean().item()
print(f"grad_w2: {grad_w2}, grad_w1: {grad_w1}")

optimizer.step()
optimizer.zero_grad()

print("Step 2")
out = head.target_time_mlp(dummy_s_emb)
loss = out.sum()
loss.backward()

grad_w2 = head.target_time_mlp[-1].weight.grad.abs().mean().item()
grad_w1 = head.target_time_mlp[0].weight.grad.abs().mean().item()
print(f"grad_w2: {grad_w2}, grad_w1: {grad_w1}")