"""
Data model for the riffusion API.
"""
from __future__ import annotations

import typing as T
from dataclasses import dataclass
from PIL import Image

# new input datatype better suited for text + image 2 image generation
@dataclass(frozen=True)
class Img2ImgInput:

    # Text prompt fed into a CLIP model
    text_prompt: str

    # Random seed for denoising
    seed: int

    # path to initial spectrogram as an image
    init_spectrogram: Image.Image

    # mask for spectrogram
    mask_image: T.Optional[Image.Image] = None

    # Negative text prompt to avoid (optional)
    negative_prompt: T.Optional[str] = None

    # Denoising strength (AKA strength)
    denoising: float = 0.75 

    # Classifier-free guidance strength. guidance=1 corresponds to no guidance.
    # guidance > 1 strengthens effect of guidance. guidance is necessary for effective
    # text conditioning accordign to Imagen paper
    guidance: float = 7.0

    # number of diffusion sampling steps
    ddim_steps: int = 50 

    # parameter for diffusion. 0.0 corresponds to deterministic sampling
        # eta (η) is only used with the DDIMScheduler, and should be between [0, 1]
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    ddim_eta: float = 0.0 # number

    # other
    # output directory
    # skip saving grid (skip_grid)
    # skip saving individual samples (skip_save)
    # fixed_code: if enabled, uses same starting code across all samples (dont need if doing only 1 sample)
    # n_iter (sample this often)