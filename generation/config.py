import torch
import random
from pathlib import Path
from dataclasses import dataclass

MATERIAL = [
    "aluminum",
    "stainless steel",
    "titanium",
    "carbon steel",
]

FINISH = [
    "brushed finish",
    "matte finish",
    "polished finish",
    "satin finish",
]

PROCESS = [
    "CNC-machined",
    "precision-milled",
    "machined",
]

CONDITION = [
    "with subtle machining marks",
    "with minor surface imperfections",
    "clean and new",
    "with fine tool marks",
]


def build_metal_description(seed: int) -> str:
    rng = random.Random(seed)   # <-- deterministic

    return ", ".join([
        rng.choice(MATERIAL),
        rng.choice(FINISH),
        rng.choice(PROCESS),
        rng.choice(CONDITION),
    ])


def build_prompt(base_prompt: str, seed: int) -> str:
    metal_desc = build_metal_description(seed)
    return base_prompt.replace("metallic materials", metal_desc)

@dataclass
class Img2BrepConfig:
    transformer_path: Path = "/mnt/d/model/Flux1_Kontext_dev_GGUF/flux1-kontext-dev-Q8_0.gguf"
    base_model_path: Path = "/mnt/d/model/Flux1_Kontext_dev"

    prompt: str = "Make this CAD model render look like a real CAD component photo, realistic lighting, metallic materials. Pure white background. Maintain sharp boundaries. Remove jagged edges (anti-aliasing), No jagged textures. High detail and clarity."

    pre_encode_text: bool = True
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype: torch.dtype = torch.bfloat16
    num_steps: int = 40
    seed: int = 0