import os
import torch
import yaml
from pathlib import Path
from src.evo.models.builder import build_model
from src.evo.serving.profiler import InferenceProfiler, analyze_model_stats

def load_configs(config_path="configs/train/distill.yaml"):
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / config_path
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_dummy_inputs(config, device="cuda"):
    fm_config = config.get("flowmatching", config.get("action_head", {}))
    if isinstance(fm_config, str): fm_config = {}
    state_dim = fm_config.get("state_dim", 7)
    
    # Mock exactly what server.py prepares
    images = [torch.randn(3, 448, 448, device=device, dtype=torch.float32) for _ in range(3)]
    state_input = torch.randn(1, state_dim, device=device, dtype=torch.float32)
    prompt = "A dummy instruction for the robot to execute."
    image_mask = torch.ones((1,), dtype=torch.int32, device=device)
    action_mask = torch.ones((1, fm_config.get("per_action_dim", 7)), dtype=torch.int32, device=device)
    
    return {
        "images": images,
        "image_mask": image_mask,
        "prompt": prompt,
        "state_input": state_input,
        "action_mask": action_mask
    }

def run_test(config, model_name="Student", device="cuda", num_iters=10):
    print(f"\n=========================================================")
    print(f"[{model_name.upper()}] INITIALIZATION AND PROFILING")
    print(f"=========================================================\n")
    
    # 1. Build Model
    model = build_model(config).eval().to(device)
    print(f"[OK] Model {model_name} initialized.")
    
    # 2. Count Params & Hardware MACs
    print(f"\n---> Running Parameter & Hardware MACs Analyzer for {model_name}...")
    dummy_kwargs = get_dummy_inputs(config, device)
    # Using torch.amp.autocast to mock the context logic in server.py
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        analyze_model_stats(model, input_kwargs=dummy_kwargs)
        
    # 3. Test Latency
    print(f"\n---> Benchmarking Latency for {model_name} ({num_iters} iters)...")
    
    # Context manager profiler setup
    enable_flops = os.environ.get("ENABLE_CUDA_PROFILE", "0") == "1"
    profiler = InferenceProfiler(warmup_steps=3, enable_flops_profiling=enable_flops, device=device)
    
    for i in range(num_iters):
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            with profiler:
                action = model.run_inference(**dummy_kwargs)
                
    profiler.summary()
    
    # clean up to save vram
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running tests on device: {device}")
    
    configs = load_configs()
    
    # Teacher Config
    teacher_cfg = configs.get("teacher_cfg", {})
    if teacher_cfg:
        run_test(teacher_cfg, model_name="Teacher_Evo1", device=device, num_iters=5)
    else:
        print("Teacher config not found in distill.yaml")
        
    # Student Config
    student_cfg = configs.get("model", {})
    if student_cfg:
        run_test(student_cfg, model_name="Student_Evo1", device=device, num_iters=5)
    else:
        print("Student config not found in distill.yaml")

