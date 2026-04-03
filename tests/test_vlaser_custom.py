import torch
from unittest.mock import MagicMock
from PIL import Image
import sys
import os
from pathlib import Path

# Thêm project root vào thư mục sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evo.models.evo1_custom import EVO1Custom

def test_vlaser_custom_instantiation():
    print("Testing initial instantiation setup of EVO1Custom with Vlaser...")
    config_path = "models/evo1_vlaser.yaml"
    
    # We will mock AutoModel / AutoTokenizer in InternVL3Embedder so we don't need to download weights during test
    import src.evo.models.components.vision_encoders.internvl3_embedder as embedder_module
    
    class MockLanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MagicMock()
            self.lm_head = MagicMock()
            self.layers = MagicMock()
            
    class MockInternVL3Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = MockLanguageModel()
            self.vision_model = MagicMock()
            
    embedder_module.AutoModel = MagicMock()
    embedder_module.AutoModel.from_pretrained.return_value = MockInternVL3Model()
    embedder_module.AutoTokenizer = MagicMock()
    
    try:
        model = EVO1Custom(config=config_path)
        print("EVO1Custom instantiated successfully with Vlaser configuration.")
        
        # Test shape logic
        assert model.horizon == 16, "Horizon should be 16"
        assert model.per_action_dim == 7, "Action dim should be 7"
        
        print("Success! The wrapper logic and config loading is correct.")
    except Exception as e:
        print(f"Failed to instantiate EVO1Custom: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vlaser_custom_instantiation()
