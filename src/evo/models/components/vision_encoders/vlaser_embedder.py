import torch
import torch.nn as nn
from typing import List, Union, Tuple
from PIL import Image

from src.evo.models.components.vision_encoders.internvl3_embedder import InternVL3Embedder

class VlaserEmbedder(InternVL3Embedder):
    """
    VlaserEmbedder wraps the InternVL3Embedder for Vlaser VLM.
    Since Vlaser is structurally identical to InternVL3 (Supervised Fine-Tuned),
    it inherits the tokenization and fusion mechanisms.
    """
    def __init__(self, model_name="OpenGVLab/InternVL3-2B", image_size=448, device="cuda"):
        super().__init__(model_name=model_name, image_size=image_size, device=device)