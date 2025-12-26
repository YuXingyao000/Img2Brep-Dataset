import torch
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Img2BrepConfig:
    transformer_path: Path = "/path/to/your/quantization/transformer/model/"
    base_model_path: Path = "/path/to/your/FLUX-Kontext-dev/model/"

    prompt: str = "Make this CAD model render look like a real CAD component photo, realistic lighting, metallic materials. Pure white background."

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype: torch.dtype = torch.bfloat16
    num_steps: int = 40
    seed: int = 0