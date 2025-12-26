# Natural CAD component pictures

Generate natural CAD component pictures from [ABC](https://deep-geometry.github.io/abc-dataset/)

Raw white model pictures are created by [pythonocc-core](https://github.com/tpaviot/pythonocc-core)

The image generation model is [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev). Specifically, the [GGUF 8-bit quantization](https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF).

## Installation
```
conda create -n flux python=3.10
conda activate flux
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Get Started
Check the `generate.py` file: 

1. Set the data path.
2. Change the `config` if needed.
3. Then,  `python -m generation.generate`

## Citation

If you use this project in academic work, please cite the following resources.

### pythonocc-core

This project uses `pythonocc-core` for CAD geometry processing and rendering.

> Paviot, T. (2022). *pythonocc*. Zenodo.  
> https://doi.org/10.5281/zenodo.3605364

### FLUX.1-Kontext

This project uses the FLUX.1-Kontext diffusion model for image generation.

> Black Forest Labs et al. (2025). *FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space*.  
> arXiv:2506.15742  
> https://arxiv.org/abs/2506.15742