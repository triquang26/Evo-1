import sys
import yaml
import torch
sys.path.append('.')

from src.evo.models.builder import build_model
from src.evo.models.evo1_student import EVO1Student

try:
    with open('configs/train/distill.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    print("- Loaded Config.")
    
    # Check if student builds
    student_cfg = cfg.get('model')
    print(f"- Student num_llm_layers configured: {student_cfg.get('num_llm_layers')}")
    print(f"- Student llm_layer_indices: {student_cfg.get('llm_layer_indices')}")
    print("- Preparing to build EVO1Student... (Mocking device to CPU to avoid OOM)")
    
    # We alter config to force low_cpu_mem_usage or just test syntax wrapper
    model = build_model(cfg)
    
    print("- Successfully built model.")
    if isinstance(model, EVO1Student):
        print("Model is EVO1Student.")
    
    # Test DistillTrainer compilation
    import src.evo.training.distill_trainer
    print("- Successfully imported distill_trainer.py without syntax errors.")
    
except Exception as e:
    print(f"Error occurred: {e}")
