import torch
from types import SimpleNamespace
from src.evo.models.components.action_heads.snapflow_action_head import SnapFlowActionHead

def test_snapflow_action_head():
    # Cố định random seed để loại bỏ yếu tố ngẫu nhiên khác
    torch.manual_seed(42)

    B = 2
    horizon = 16
    per_action_dim = 7
    embed_dim = 896
    
    config = SimpleNamespace(
        embed_dim=embed_dim,
        hidden_dim=1024,
        action_dim=horizon * per_action_dim,
        horizon=horizon,
        per_action_dim=per_action_dim,
        state_dim=7,
        state_hidden_dim=1024,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        num_inference_timesteps=1,
        num_categories=1
    )
    
    print("Khởi tạo SnapFlowActionHead...")
    head = SnapFlowActionHead(config=config)
    head.eval()

    # Tạo mock tensors
    fused_tokens = torch.randn(B, 32, embed_dim) # seq_len giả định 32
    state = torch.randn(B, 7)
    actions_gt = torch.randn(B, horizon, per_action_dim)
    action_mask = torch.ones(B, horizon, per_action_dim)
    
    try:
        # CỐ ĐỊNH NOISE ĐỂ SO SÁNH GIỮA 2 ĐƯỜNG TRUYỀN
        noise_fixed = torch.randn_like(actions_gt)

        # 1. Kiểm tra forward pass (Loss computation flow)
        print("Testing Forward Pass (Training Flow)...")
        # Không có s (Flow matching thường)
        t = torch.tensor([0.5, 0.8])
        pred1, noise1 = head(fused_tokens, state=state, actions_gt=actions_gt, action_mask=action_mask, t=t, noise=noise_fixed)
        assert pred1.shape == (B, horizon * per_action_dim)
        
        # Có s (Shortcut Student flow)
        s = torch.tensor([0.0, 0.0])
        pred2, noise2 = head(fused_tokens, state=state, actions_gt=actions_gt, action_mask=action_mask, t=t, s=s, noise=noise_fixed)
        assert pred2.shape == (B, horizon * per_action_dim)
        
        print("✓ Forward pass thành công! Tensor shapes chuẩn.")
        
        # 2. Kiểm tra Zero-init của Target-Time Embedding
        print("Testing Zero-Initialization Requirement...")
        diff = torch.abs(pred1 - pred2).max()
        assert diff < 1e-6, f"Lỗi Zero-init! Sự sai khác là {diff.item()}"
        print("✓ Zero-initialization thành công! Dự đoán tại s=0 hoàn toàn tương đương với khi không có s.")
        
        # 3. Kiểm tra Inference pass (1-NFE)
        print("Testing 1-NFE Inference Bypass...")
        inferred_action = head.get_action(fused_tokens, state=state, action_mask=action_mask, s=0.0, t=1.0)
        assert inferred_action.shape == (B, horizon * per_action_dim)
        print("✓ 1-NFE Inference thành công!")
        
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_snapflow_action_head()
