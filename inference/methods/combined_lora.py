import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F

import os
import tqdm
import numpy as np
from pathlib import Path
import hashlib
import itertools
import math

from datasets.utils import build_data_loader
from .diffusion_classifier import DiffusionClassifier
from .utils_diffusion import get_sd_model, get_scheduler_config, MODEL_IDS
from .utils import dict2obj
from .utils_dreambooth_lora import PromptDataset, DreamBoothDataset, get_display_images
from .lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
    safetensors_available,
    save_lora_weight,
    save_safeloras,
    tune_lora_scale,
    monkeypatch_or_replace_lora
)
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer


class CombinedLora(DiffusionClassifier):
    def __init__(self, config, dataset, device):
        super().__init__(config, dataset, device)
        text_lora_dir = f"{config['lora_dir']}/text_{config['iteration']}.pt"
        unet_lora_dir = f"{config['lora_dir']}/unet_{config['iteration']}.pt"
        self.lora_token = config['lora_token']

        print(f"Replacing text lora with {text_lora_dir}...")
        monkeypatch_or_replace_lora(self.text_encoder, torch.load(text_lora_dir), target_replace_module=["CLIPAttention"])
        print(f"Setting text lora scale {config['text_lora_scale']}...")
        tune_lora_scale(self.text_encoder, config["text_lora_scale"])
        
        print(f"Replacing unet lora with {unet_lora_dir}...")
        monkeypatch_or_replace_lora(self.unet, torch.load(unet_lora_dir))
        print(f"Setting unet lora scale {config['unet_lora_scale']}...")
        tune_lora_scale(self.unet, config["unet_lora_scale"])

        print("regenerating text embeddings...")
        self.construct_text_embeddings()
    
    def get_classname(self, dataset_classname):
        cname = dataset_classname.replace('_', ' ')
        if not hasattr(self, 'lora_token') or self.lora_token is None:
            return cname
        else:
            return f"{self.lora_token} {cname}" # a photo of a hta 737, a type of aircraft.
    
    '''
    def eval_error_custom(self, unet, scheduler, latent, all_noise, ts, noise_idxs,
                text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2', device='cpu'):
        assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
        # print("Using custom eval error...")
        pred_errors = torch.zeros(len(ts), device='cpu')
        idx = 0
        with torch.inference_mode():
            for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
                batch_ts = torch.tensor(ts[idx: idx + batch_size])
                batch_ts[batch_ts > 550] = 350
                assert (batch_ts<0).sum() == 0
                noise = all_noise[noise_idxs[idx: idx + batch_size]]
                noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                                noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
                t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
                text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
                noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
                if loss == 'l2':
                    error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                elif loss == 'l1':
                    error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                elif loss == 'huber':
                    error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                else:
                    raise NotImplementedError
                pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
                idx += len(batch_ts)
        return pred_errors
    '''
    