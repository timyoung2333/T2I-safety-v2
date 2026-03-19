import torch
from PIL import Image
import numpy as np
from safetensors.torch import load_file
import random
import pandas as pd
import os
import sys
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import json
import numpy as np
import pandas as pd
from PIL import Image
from argparse import ArgumentParser
import random
from collections import defaultdict
import time
import torch.nn.functional as F
import random
from datasets import load_dataset
from transformers import AutoTokenizer



class Configuration:
    def __init__(self, *args, **kwargs) -> None:
        args = args[0]

        self.ckpt = args["ckpt"]
        self.save_path = args["save_path"]
        self.data_path = args["data_path"]
       
        



def main(config: Configuration):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = config.ckpt
    save_path = config.save_path
    data_path = config.data_path

    if not os.path.exists(save_path):
            os.makedirs(save_path)

    text_encoder = CLIPTextModel.from_pretrained(ckpt)
    vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="vae", use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="tokenizer", use_fast=False) 
    unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="unet", use_safetensors=True)
    scheduler = DDIMScheduler.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder="scheduler", use_safetensors=True)

    model = StableDiffusionPipeline(
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None
    )

    model = model.to(device)
    




    df = pd.read_csv(data_path)

    print(len(df))

    column_to_use = "adv_prompt" if "adv_prompt" in df.columns else "prompt"
    
    for i, row in df.iterrows():
        
        prompt = str(row[column_to_use])  
        seed = 7867 if column_to_use == "adv_prompt" else row.evaluation_seed
        
        print(prompt)
        print(seed)
        
        generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"Generating image {i}...")

        img = model(prompt=prompt, num_inference_steps=50, generator=generator).images[0]

        img.save(f"{save_path}/{i}.png")
      

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model with target concept erased")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save generated images")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV file ")

    args = vars(parser.parse_args())
    config = Configuration(args)
    main(config) 
