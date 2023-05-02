# some helper functions needed
import matplotlib.pyplot as plt
import numpy as np
import torch 
import einops 
from pytorch_lightning import seed_everything
import cv2
import os
from PIL import Image
import numpy as np
import random 

from cldm.model import create_model, load_state_dict

# get cnet model loaded into state dict
def get_model(mdl_path, mdl_config='./models/cldm_v15.yaml'):
    model = create_model(mdl_config).cpu()
    model.load_state_dict(load_state_dict(mdl_path, location='cuda'))
    model = model.cuda()
    return model

# image helper functions
def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
    
def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

# helper function to sample the model
def sample_ddim(control, prompt, model, ddim_sampler,
                guess_mode = False,
                strength = 1,
                scale = 9,
                eta = 0,
                num_samples = 1,
                ddim_steps = 50,
                seed = -1,
                control_lims = [0.,255.],
                image_resolution = 512,
                log_every_t=100):

    with torch.no_grad():
        # preprocessing
        if (np.max(control) <= 1) and (np.min(control) >= 0):
            control = np.uint8(control * 255)
        control = resize_image(HWC3(control), image_resolution)
        H, W, C = control.shape
        # rescale back to desired control input range
        control = np.float64(control) / 255.0 * (control_lims[1] - control_lims[0]) + control_lims[0]
        control = torch.from_numpy(control).float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # to save memory
        model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([""] * num_samples)]}
        shape = (4, H // 8, W // 8)

        # to save memory
        model.low_vram_shift(is_diffusing=True)

        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  
        
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale, log_every_t=log_every_t,
                                                        unconditional_conditioning=un_cond)

        # to save memory
        model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

        return results, intermediates