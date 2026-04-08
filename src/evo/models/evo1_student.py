import torch
import torch.nn as nn
from typing import Union
from pathlib import Path
from src.evo.models.evo1 import EVO1
from src.evo.models.components.vision_encoders.internvl3_embedder import InternVL3Embedder
from src.evo.models.components.action_heads.flow_matching import FlowmatchingActionHead
from types import SimpleNamespace

class EVO1Student(EVO1):
    def __init__(self, config: Union[str, Path, dict]):
        super().__init__(config)
        
        model_cfg = self.config.get("model", self.config)
        vlm_name = model_cfg.get("vision_encoder", "OpenGVLab/InternVL3-1B")
        num_llm_layers = model_cfg.get("num_llm_layers", 4)
        llm_layer_indices = model_cfg.get("llm_layer_indices", None)
        
        self.embedder = InternVL3Embedder(
            model_name=vlm_name, 
            device=self._device, 
            num_llm_layers=num_llm_layers,
            llm_layer_indices=llm_layer_indices
        )
        
       
        teacher_vlm_dim = model_cfg.get("teacher_vlm_dim", 2048)
        student_vlm_dim = 2048 
        
        if teacher_vlm_dim != student_vlm_dim:
            self.kd_projector = nn.Linear(student_vlm_dim, teacher_vlm_dim).to(self._device)
        else:
            self.kd_projector = nn.Identity().to(self._device)

    def get_projected_vl_embeddings(self, *args, **kwargs):
        """Returns the VL embeddings projected to the teacher's latent space for Feature KD."""
        student_embeds = self.get_vl_embeddings(*args, **kwargs)
        return self.kd_projector(student_embeds)
