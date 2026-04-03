from src.evo.models.evo1 import EVO1
from src.evo.models.evo1_custom import EVO1Custom

def build_model(config):
    model_type = config.get("model", {}).get("type", "evo1").lower()
    
    if model_type == "evo1":
        return EVO1(config)
    elif model_type == "evo1_custom":
        return EVO1Custom(config)
    
    raise ValueError(f"Unknown model type: {model_type}")
