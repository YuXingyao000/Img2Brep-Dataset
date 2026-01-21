import os
import torch
import ray
import numpy as np
import traceback
from math import ceil
from pathlib import Path
from PIL import Image
from .config import Img2BrepConfig, build_prompt
from .pipeline import Img2BrepPipeline

# -----------------------------
# Config
# -----------------------------
IMG_ROOT = Path("/mnt/d/data/abc_v2_natural_AA_Sketch2")
OUT_DIR = Path("/mnt/d/data/abc_v2_natural_AA_Sketch2_generated")
NUM_SERVER = 0
TOTAL_NUM_SERVER = 5
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def collect_server_folders(img_root: Path, num_server: int, total_num_server: int) -> list[Path]:
    all_folders = sorted([p for p in img_root.iterdir() if p.is_dir()])
    num_folder_per_server = len(all_folders) // total_num_server
    folder_start = num_server * num_folder_per_server

    if num_server == total_num_server - 1:
        return all_folders[folder_start:]
    return all_folders[folder_start : folder_start + num_folder_per_server]


def chunk_list(xs: list[Path], n: int) -> list[list[Path]]:
    """Split xs into n nearly-equal chunks (n must be >= 1)."""
    if n <= 1:
        return [xs]
    k = ceil(len(xs) / n)
    return [xs[i * k : (i + 1) * k] for i in range(n)]

def load_img_from_npz(config, img_folder: Path) -> np.ndarray:
    data_file = img_folder / "data.npz"

    arr = np.load(data_file)["svr_imgs"]
            
    hashseed = hash(img_folder.stem) % (2**32)
    rng = np.random.default_rng(seed=hashseed + config.seed)
    idx = rng.integers(64, 128)
    img_data = arr[idx]
    return img_data

def load_img(img_folder: Path) -> np.ndarray:
    img_file = img_folder / "sketch.png"
    img_data = np.array(Image.open(img_file).convert("RGB"))
    return img_data

# @ray.remote(num_gpus=1)
def generate(config_dict: dict, folder_chunk: list[str], chunk_id: int):
    """
    Pass strings/paths as strings to reduce serialization surprises.
    Build config inside worker to avoid pickling issues.
    """
    try:
        config = Img2BrepConfig()
        for k, v in config_dict.items():
            setattr(config, k, v)

        pipe = Img2BrepPipeline(config)

        out_root = Path(OUT_DIR.as_posix() + f"_{NUM_SERVER}")

        for i, folder_str in enumerate(folder_chunk):
            img_folder = Path(folder_str)
            img_data = load_img(img_folder)

            # random_prompt = build_prompt(config.prompt, seed=hashseed)
            # config.prompt = random_prompt
            generated_img: Image.Image = pipe(img_data)

            output_path = out_root / img_folder.stem / "natural.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            generated_img.save(output_path)

        return (True, chunk_id, "")

    except Exception as e:
        return (False, chunk_id, traceback.format_exc())




if __name__ == "__main__":
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert num_gpus > 0, "No CUDA GPUs visible."
    ray.init(num_gpus=num_gpus)
    
    #########################
    # Step1: Initialization #
    #########################
    config = Img2BrepConfig()
    config_dict = {}
    config_dict["pre_encode_text"] = True
    # Example:
    # config_dict["base_model_path"] = "/path/to/base"
    
    ######################
    # Step2: Load images #
    ######################
    server_folders = collect_server_folders(IMG_ROOT, NUM_SERVER, TOTAL_NUM_SERVER)
    
    #############################
    # Step3: Split into batches #
    #############################
    chunks = chunk_list(server_folders, num_gpus)
    chunks = [[p.as_posix() for p in chunk] for chunk in chunks]
    
    # ##########################
    # # Step4: Multiprocessing #
    # ##########################
    generate(config_dict, chunks[0], 0)  # for debugging
    refs = [generate.remote(config_dict, chunks[i], i) for i in range(len(chunks))]
    outs = ray.get(refs)    
    
    for ok, chunk_id, err in outs:
        if not ok:
            print(f"[ERROR] chunk {chunk_id} failed:\n{err}")
        else:
            print(f"[OK] chunk {chunk_id} done")
        
    