import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

from .config import Img2BrepConfig

class Img2BrepPipeline:
    def __init__(self, config: Img2BrepConfig):
        self.config = config
        self.pipeline = self._init_pipe()
        output = self._preencode_prompt(self.pipeline, self.config.pos_prompt) if self.config.pre_encode_text else (None, None, None)
        (self.prompt_embeds, self.pooled_prompt_embeds, _) = output
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
            # img = img.resize((224 * 4, 224 * 4), resample=Image.Resampling.LANCZOS)
            # img = img.filter(ImageFilter.GaussianBlur(1.2))
            # edges = Img2BrepPipeline.extract_edges(img)
            # smoothed = Img2BrepPipeline.sdf_antialias_edges_bold(edges, thresh=220, width_px=1.8, output_mode="RGB")
            
            out = self.pipeline(
                image=img,
                prompt= self.config.pos_prompt if not self.config.pre_encode_text else None,
                prompt_embeds=self.prompt_embeds if self.config.pre_encode_text else None,
                pooled_prompt_embeds=self.pooled_prompt_embeds if self.config.pre_encode_text else None,
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
    
    @staticmethod
    def extract_edges(img: Image.Image) -> Image.Image:
        gray = np.array(img.convert("L"))
        edge_mask = gray < 100
        out = np.ones(edge_mask.shape, dtype=np.uint8) * 255
        out[edge_mask] = 0
        return Image.fromarray(out, mode="L").convert("RGB")
    
    @staticmethod
    def sdf_antialias_edges_bold(
        edge_img: Image.Image,
        *,
        thresh: int = 220,
        width_px: float = 1.2,
        bold_px: float = 1.0,
        output_mode: str = "RGB",
    ) -> Image.Image:
        """
        SDF-based anti-aliasing + stroke bolding for black-on-white edge images.

        Parameters
        ----------
        edge_img : PIL.Image
            Input image (black edges on white background).
        thresh : int
            Grayscale threshold to classify ink pixels.
        width_px : float
            Edge softness (anti-aliasing width), typical 0.8–2.0.
        bold_px : float
            Stroke expansion radius in pixels (0.5–3.0 typical).
        output_mode : str
            "L" or "RGB".

        Returns
        -------
        PIL.Image
            Anti-aliased, bolded edge image.
        """

        # 1) grayscale
        g = np.array(edge_img.convert("L"), dtype=np.uint8)

        # 2) ink mask (True where black stroke exists)
        ink = g < int(thresh)

        if not ink.any():
            return Image.new(output_mode, edge_img.size, 255)

        # 3) distance to nearest ink pixel
        dist = distance_transform_edt(~ink).astype(np.float32)

        # 4) SDF-based coverage with bolding
        #    dist - bold_px shifts the edge outward
        d = dist - float(bold_px)

        alpha = np.clip(1.0 - d / float(width_px), 0.0, 1.0)

        # Optional perceptual sharpening (smoothstep)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        # 5) composite: black ink on white background
        out = (255.0 * (1.0 - alpha) + 0.5).astype(np.uint8)

        return Image.fromarray(out, mode="L").convert(output_mode)