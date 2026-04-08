#!/usr/bin/env python3
import sys
import os
import atexit
import torch

# Ensure Evo-1 root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evo.serving.profiler import InferenceProfiler, analyze_model_stats
from lerobot.scripts.lerobot_eval import eval_main
from lerobot.policies import factory

# Save the original make_policy before patching
original_make_policy = factory.make_policy

# Global profiler tracking for summary at exit
global_profiler = None

def summary_at_exit():
    if global_profiler is not None:
        global_profiler.summary()

atexit.register(summary_at_exit)

def make_profiled_policy(cfg, env_cfg, rename_map):
    global global_profiler
    
    # Create the original policy
    policy = original_make_policy(cfg, env_cfg, rename_map)
    policy.eval()
    
    # 1. Print Model Statistics exactly like Evo-1
    print("\n[Evo-1 Profiler Wrapper] Loading Model Stats before evaluation...")
    analyze_model_stats(policy, input_kwargs=None)
    
    device = str(policy.device) if hasattr(policy, "device") else "cuda"
    enable_flops = os.environ.get("ENABLE_CUDA_PROFILE", "0") == "1"
    
    # 2. Instantiate Evo-1's InferenceProfiler
    global_profiler = InferenceProfiler(warmup_steps=3, enable_flops_profiling=enable_flops, device=device)
    
    # 3. Monkey-patch the select_action method to route through the profiler context manager
    original_select_action = policy.select_action
    
    def profiled_select_action(batch):
        with global_profiler:
            with torch.inference_mode():
                # Note: lerobot_eval.py already does inference_mode, but nested is fine
                return original_select_action(batch)
                
    policy.select_action = profiled_select_action
    
    return policy

from lerobot.scripts import lerobot_eval as lerobot_eval_module

# Override lerobot's make_policy cleanly
lerobot_eval_module.make_policy = make_profiled_policy


if __name__ == "__main__":
    # We call eval_main() which automatically parses cli args exactly like `lerobot-eval` 
    eval_main()
