"""
reference : https://github.com/fmp453/few-shot-erasing/blob/main/utils.py
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, Union



from diffusers import (
    AutoencoderKL, 
    PNDMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler

stable_diffusion_versions = {
    "14": "CompVis/stable-diffusion-v1-4",
    "15": "runwayml/stable-diffusion-v1-5",
}

clip_versions = {
    "oa": "openai/clip-vit-large-patch14",
    "oa-336": "openai/clip-vit-large-patch14-336",
}




def _load_models(clip_version, stable_diffusion_version):
    clip_version = clip_versions[clip_version]
    stable_diffusion_version = stable_diffusion_versions[stable_diffusion_version]

    tokenizer = CLIPTokenizer.from_pretrained(clip_version)
    text_encoder = CLIPTextModel.from_pretrained(clip_version, low_cpu_mem_usage=False)
    vae = AutoencoderKL.from_pretrained(stable_diffusion_version, subfolder="vae",)
    unet = UNet2DConditionModel.from_pretrained(stable_diffusion_version, subfolder="unet")
    noise_scheduler = DDIMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="scheduler", use_safetensors=True)

    return tokenizer, text_encoder, vae, unet, noise_scheduler

def _load_models_from_local(clip_version, stable_diffusion_version):
    if not(clip_version in clip_versions.keys() and stable_diffusion_version in stable_diffusion_versions.keys()):
        raise ValueError(f"{clip_version} or {stable_diffusion_version} is not valid.")
    
    tokenizer = CLIPTokenizer.from_pretrained(clip_versions[clip_version])
    text_encoder = CLIPTextModel.from_pretrained(f"models/clip/{clip_version}/text_encoder")
    vae = AutoencoderKL.from_pretrained(f"models/diffusion/{stable_diffusion_version}/vae")
    unet = UNet2DConditionModel.from_pretrained(f"models/diffusion/{stable_diffusion_version}/unet")
    noise_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True, steps_offset=1)

    return tokenizer, text_encoder, vae, unet, noise_scheduler


def load_models_from_local_optioned_path(text_encoder_path:str, tokenizer_version:str="openai/clip-vit-large-patch14"):
    
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_version)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
    noise_scheduler = DDIMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="scheduler", use_safetensors=True)

    return tokenizer, text_encoder, noise_scheduler

def load_models(clip_version, stable_diffusion_version, use_local):
    if use_local:
        return _load_models_from_local(clip_version, stable_diffusion_version)
    else:
        return _load_models(clip_version, stable_diffusion_version)

def get_optimizer(
        parameters, 
        optimizer_name:str, 
        lr:float, 
        betas:Tuple[float, float]=(0.9, 0.99), 
        weight_decay:float=1e-2, 
        eps:float=1e-6
    ):
    optim_name = optimizer_name.lower()
    
    if optim_name == "adamw":
        return optim.AdamW(parameters, lr, betas, eps, weight_decay)

    elif optim_name == "adam":
        return optim.Adam(parameters, lr, betas, eps, weight_decay)
    
    elif optim_name == "adagrad":
        return optim.Adagrad(parameters, lr, weight_decay=weight_decay, eps=eps)

    elif optim_name == "adadelta":
        return optim.Adadelta(parameters, lr, eps=eps, weight_decay=weight_decay)

    raise ValueError(f"not match optimizer name : {optimizer_name}")

def plot_loss(history:Dict, save_path:str):
    df = pd.DataFrame(history)
    df.to_csv(f"{save_path}/loss.csv", index=False)
    plt.figure()
    df.plot()
    plt.savefig(f"{save_path}/loss.png")
    plt.close('all')



def freeze_and_unfreeze_text_encoder(text_encoder, method="all"):
    if method == "all":
        return text_encoder
    
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    mlps = False
    final_attn = False
    attns = False
    first_attn = False

    if method == "mlp-only":
        mlps = True
    elif method == "attn-only":
        attns = True
    elif method == "mlp-attn":
        mlps = True
        attns = True
    elif method == "mlp-final-attn":
        mlps = True
        final_attn = True
        
    elif method == "first-attn":
        first_attn = True
        mlps = True
        

    elif method == "mix-attn":
        mlps = True
        final_attn = True
        first_attn = True

    elif method == "final-attn":
       
        final_attn = True

    for param_name, module in text_encoder.named_modules():
        if mlps and "0.mlp.fc" in param_name and "10.mlp.fc" not in param_name:
            print(param_name)
            for param in module.parameters():
                param.requires_grad = True
        
        if attns and ".self_attn." in param_name:
            print(param_name)
            for param in module.parameters():
                param.requires_grad = True
        
        
        if final_attn and "11.self_attn." in param_name:
            print(param_name)
            for param in module.parameters():
                param.requires_grad = True       
        
        if first_attn and "0.self_attn." in param_name and "10.self_attn." not in param_name:
            print(param_name)
            for param in module.parameters():
                param.requires_grad = True

        
        
    return text_encoder

