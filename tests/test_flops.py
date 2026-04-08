import torch
from src.evo.serving.profiler import InferenceProfiler
from src.evo.models.builder import build_model

def get_dummy_inputs(device="cuda"):
    images = [torch.randn(3, 448, 448, device=device, dtype=torch.bfloat16) for _ in range(3)]
    state_input = torch.randn(1, 7, device=device, dtype=torch.bfloat16)
    prompt = "A dummy instruction."
    image_mask = torch.ones((1,), dtype=torch.int32, device=device)
    action_mask = torch.ones((1, 7), dtype=torch.int32, device=device)
    return {"images": images, "image_mask": image_mask, "prompt": prompt, "state_input": state_input, "action_mask": action_mask}

model = build_model({"model": {"type": "evo1_student", "vision_encoder": "OpenGVLab/InternVL3-1B", "num_llm_layers": 1, "action_head": "flowmatching", "flowmatching": {"state_dim": 7, "per_action_dim": 7, "num_layers": 4, "embed_dim": 896, "hidden_dim": 1024, "num_heads": 8, "dropout": 0.1, "num_inference_timesteps": 50, "action_horizon": 16}}}).to("cuda")

prof = InferenceProfiler(warmup_steps=0, enable_flops_profiling=True, device="cuda")
with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    with prof:
        model.run_inference(**get_dummy_inputs())
prof.summary()
