"""
Adapted from the Hugging Face Space:

  awacke1/Image-to-Line-Drawings
  https://huggingface.co/spaces/awacke1/Image-to-Line-Drawings

Original author: awacke1
Original project: Image-to-Line-Drawings (Hugging Face Space)

Used/referenced under the permissions granted by Hugging Face Hub. See the
original repository for license details and additional attribution.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageOps, ImageChops
import torchvision.transforms as transforms
import os
import ray
from math import ceil
from pathlib import Path

norm_layer = nn.InstanceNorm2d
IMG_ROOT = Path("/mnt/d/data/abc_v2_npz")
OUT_DIR = Path("/mnt/d/data/abc_v2_natural_AA_Sketch2")
NUM_SERVER = 0
TOTAL_NUM_SERVER = 5
MODEL1_PATH = "model.pth"
MODEL2_PATH = "model2.pth"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# 🧱 Building block for the generator
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features) ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# 🎨 Generator model for creating line drawings
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()
        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)
        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)
        # Residual blocks
        model2 = []
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)
        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)
        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]
        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)
        return out

def apply_filter(line_img, filter_name, original_img):
    if filter_name == "Standard":
        return line_img
        
    # Convert line drawing to grayscale for most operations
    line_img_l = line_img.convert('L')

    # --- Standard Image Filters ---
    if filter_name == "Invert": return ImageOps.invert(line_img_l)
    if filter_name == "Blur": return line_img.filter(ImageFilter.GaussianBlur(radius=3))
    if filter_name == "Sharpen": return line_img.filter(ImageFilter.SHARPEN)
    if filter_name == "Contour": return line_img_l.filter(ImageFilter.CONTOUR)
    if filter_name == "Detail": return line_img.filter(ImageFilter.DETAIL)
    if filter_name == "EdgeEnhance": return line_img_l.filter(ImageFilter.EDGE_ENHANCE)
    if filter_name == "EdgeEnhanceMore": return line_img_l.filter(ImageFilter.EDGE_ENHANCE_MORE)
    if filter_name == "Emboss": return line_img_l.filter(ImageFilter.EMBOSS)
    if filter_name == "FindEdges": return line_img_l.filter(ImageFilter.FIND_EDGES)
    if filter_name == "Smooth": return line_img.filter(ImageFilter.SMOOTH)
    if filter_name == "SmoothMore": return line_img.filter(ImageFilter.SMOOTH_MORE)

    # --- Tonal Adjustments ---
    if filter_name == "Solarize": return ImageOps.solarize(line_img_l)
    if filter_name == "Posterize1": return ImageOps.posterize(line_img_l, 1)
    if filter_name == "Posterize2": return ImageOps.posterize(line_img_l, 2)
    if filter_name == "Posterize3": return ImageOps.posterize(line_img_l, 3)
    if filter_name == "Posterize4": return ImageOps.posterize(line_img_l, 4)
    if filter_name == "Equalize": return ImageOps.equalize(line_img_l)
    if filter_name == "AutoContrast": return ImageOps.autocontrast(line_img_l)
    if filter_name == "Binary": return line_img_l.convert('1')

    # --- Morphological Operations (Thick/Thin) ---
    if filter_name == "Thick1": return line_img_l.filter(ImageFilter.MinFilter(3))
    if filter_name == "Thick2": return line_img_l.filter(ImageFilter.MinFilter(5))
    if filter_name == "Thick3": return line_img_l.filter(ImageFilter.MinFilter(7))
    if filter_name == "Thin1": return line_img_l.filter(ImageFilter.MaxFilter(3))
    if filter_name == "Thin2": return line_img_l.filter(ImageFilter.MaxFilter(5))
    if filter_name == "Thin3": return line_img_l.filter(ImageFilter.MaxFilter(7))

    # --- Colorization (On White Background) ---
    colors_on_white = {"RedOnWhite": "red", "OrangeOnWhite": "orange", "YellowOnWhite": "yellow", "GreenOnWhite": "green", "BlueOnWhite": "blue", "PurpleOnWhite": "purple", "PinkOnWhite": "pink", "CyanOnWhite": "cyan", "MagentaOnWhite": "magenta", "BrownOnWhite": "brown", "GrayOnWhite": "gray"}
    if filter_name in colors_on_white:
        return ImageOps.colorize(line_img_l, black=colors_on_white[filter_name], white="white")

    # --- Colorization (On Black Background) ---
    colors_on_black = {"WhiteOnBlack": "white", "RedOnBlack": "red", "OrangeOnBlack": "orange", "YellowOnBlack": "yellow", "GreenOnBlack": "green", "BlueOnBlack": "blue", "PurpleOnBlack": "purple", "PinkOnBlack": "pink", "CyanOnBlack": "cyan", "MagentaOnBlack": "magenta", "BrownOnBlack": "brown", "GrayOnBlack": "gray"}
    if filter_name in colors_on_black:
        return ImageOps.colorize(line_img_l, black=colors_on_black[filter_name], white="black")

    # --- Blending Modes with Original Image ---
    line_img_rgb = line_img.convert('RGB')
    if filter_name == "Multiply": return ImageChops.multiply(original_img, line_img_rgb)
    if filter_name == "Screen": return ImageChops.screen(original_img, line_img_rgb)
    if filter_name == "Overlay": return ImageChops.overlay(original_img, line_img_rgb)
    if filter_name == "Add": return ImageChops.add(original_img, line_img_rgb)
    if filter_name == "Subtract": return ImageChops.subtract(original_img, line_img_rgb)
    if filter_name == "Difference": return ImageChops.difference(original_img, line_img_rgb)
    if filter_name == "Darker": return ImageChops.darker(original_img, line_img_rgb)
    if filter_name == "Lighter": return ImageChops.lighter(original_img, line_img_rgb)
    if filter_name == "SoftLight": return ImageChops.soft_light(original_img, line_img_rgb)
    if filter_name == "HardLight": return ImageChops.hard_light(original_img, line_img_rgb)
    
    # --- Texture ---
    if filter_name == "Noise":
        img_array = np.array(line_img_l.convert('L'))
        noise = np.random.randint(-20, 20, img_array.shape, dtype='int16')
        noisy_array = np.clip(img_array.astype('int16') + noise, 0, 255).astype('uint8')
        return Image.fromarray(noisy_array)

    return line_img # Default fallback

@ray.remote(num_gpus=1)
class LineartWorker:
    def __init__(self):
        self.model1 = Generator(3, 1, 3)
        self.model1.load_state_dict(torch.load(MODEL1_PATH, map_location="cpu", weights_only=True))
        self.model1.eval()

        self.model2 = Generator(3, 1, 3)
        self.model2.load_state_dict(torch.load(MODEL2_PATH, map_location="cpu", weights_only=True))
        self.model2.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def predict_folder(self, input_img_path: str, line_style: str, seed: int = 0) -> bool:
        input_img_path = Path(input_img_path)
        data_file = input_img_path / "data.npz"

        hashseed = hash(input_img_path.stem) % (2**32)
        rng = np.random.default_rng(seed=hashseed + seed)
        idx = rng.integers(64, 128)

        arr = np.load(data_file)["svr_imgs"]
        img_data = arr[idx]
        original_img = Image.fromarray(img_data).convert("RGB")
        original_size = original_img.size

        input_tensor = self.transform(original_img).unsqueeze(0)

        with torch.no_grad():
            output = self.model2(input_tensor) if line_style == "Simple Lines" else self.model1(input_tensor)

        line_low = transforms.ToPILImage()(output.squeeze().cpu().clamp(0, 1))
        line_full = line_low.resize(original_size, Image.Resampling.BICUBIC)

        out_dir = OUT_DIR / input_img_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        line_full.save(out_dir / "sketch.png")
        return True
    
if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0

    # IMPORTANT: move Ray temp off /tmp to avoid your earlier OutOfDiskError
    ray.init(num_gpus=num_gpus, _temp_dir="/mnt/d/ray_tmp")

    all_folders = sorted([p for p in IMG_ROOT.iterdir() if p.is_dir()])

    # one actor per GPU
    workers = [LineartWorker.remote() for _ in range(num_gpus)]

    MAX_IN_FLIGHT = num_gpus * 4
    pending = []
    for i, folder in enumerate(all_folders):
        w = workers[i % num_gpus]
        pending.append(w.predict_folder.remote(str(folder), "Simple Lines", 0))

        if len(pending) >= MAX_IN_FLIGHT:
            ray.get(pending)
            pending.clear()

    if pending:
        ray.get(pending) 
