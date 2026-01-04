import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

from .config import Img2BrepConfig

class Img2BrepPipeline:
    def __init__(self, config: Img2BrepConfig):
        self.config = config
        self.pipeline = self._init_pipe()
        self.prompt_embeds, self.pooled_prompt_embeds, _ = self._preencode_prompt(self.pipeline, self.config.prompt) if self.config.pre_encode_text else None, None, None
        self.gen = torch.Generator(device=self.config.device).manual_seed(self.config.seed)
    
    def _init_pipe(self):
        transformer = FluxTransformer2DModel.from_single_file(
            self.config.transformer_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=self.config.dtype),
            torch_dtype=self.config.dtype,
            config=str(self.config.base_model_path),
            subfolder="transformer",
            local_files_only=True
        )

        pipe = FluxKontextPipeline.from_pretrained(
            self.config.base_model_path,
            transformer=transformer,
            torch_dtype=self.config.dtype,
            local_files_only=True
        )

        pipe.transformer.config.in_channels = 64

        pipe.vae.to(device=self.config.device, dtype=self.config.dtype)
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        pipe.enable_model_cpu_offload()

        return pipe

    def _preencode_prompt(self,pipe: FluxKontextPipeline, prompt: str):
        with torch.inference_mode():
            pipe.text_encoder.to(device=self.config.device, dtype=self.config.dtype)
            pipe.text_encoder_2.to(device=self.config.device, dtype=self.config.dtype)

            pe, ppe, tids = pipe.encode_prompt(
                prompt=prompt,
                device=pipe._execution_device,
            )

            pe = pe.to(device=self.config.device, dtype=self.config.dtype)
            ppe = ppe.to(device=self.config.device, dtype=self.config.dtype)

            pipe.text_encoder.to("cpu")
            pipe.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()

            return pe, ppe, tids
    
    def __call__(self, img: np.ndarray) -> Image:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


        with torch.inference_mode():
            img = Image.fromarray(img).convert("RGB")
            out = self.pipeline(
                image=img,
                prompt= self.config.prompt if not self.config.pre_encode_text else None,
                prompt_embeds=self.prompt_embeds,
                pooled_prompt_embeds=self.pooled_prompt_embeds,
                num_inference_steps=self.config.num_steps,
                generator=self.gen,
                true_cfg_scale=1.0,
                guidance_scale=3.5,
                output_type="pt"
            )
            
            img_t = out.images[0]
            
            del out
            torch.cuda.empty_cache()
            
            img_t = torch.nan_to_num(img_t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            img_t = img_t.permute(1, 2, 0)
            img_u8 = (img_t * 255).round().to(torch.uint8).cpu().numpy()
            return Image.fromarray(img_u8)