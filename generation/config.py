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

    pos_prompt: str = "technical mechanical engineering sketch, hand-drawn pencil lineart on paper, subtle line wobble, pressure-sensitive strokes, varying line weight, slightly darker outer contour, faint construction lines, clean drafting style, light paper grain, minimal shading, precise geometry, engineer sketch, CAD model sketch"
    neg_prompt: str = "photorealistic, color, watercolor, heavy shading, messy scribbles, comic style, cartoon, thick marker, ink wash, texture overload, distorted geometry, extra parts, text, watermarks, low quality, blurry, out of focus"
    
    pre_encode_text: bool = True
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype: torch.dtype = torch.bfloat16
    num_steps: int = 40
    seed: int = 0