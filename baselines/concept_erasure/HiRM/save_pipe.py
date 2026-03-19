"""
reference : https://github.com/fmp453/few-shot-erasing/blob/main/load_save.py
"""

import os
import argparse
from transformers import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline


def text_encoder_save(model_name: str, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    text_encoder = CLIPTextModel.from_pretrained(model_name)
    text_encoder.save_pretrained(f"{save_path}/text_encoder")

    print(f"Saved text encoder model at {save_path}/text_encoder")



def pipeline_save(model_name: str, save_path: str):
    pipe = StableDiffusionPipeline.from_pretrained(model_name)
    
    os.makedirs(save_path, exist_ok=True)
    # text encoder & tokenizer
    pipe.text_encoder.save_pretrained(f"{save_path}/text_encoder")
    pipe.tokenizer.save_pretrained(f"{save_path}/tokenizer")

    
    # vae & uet & scheduler
    pipe.unet.save_pretrained(f"{save_path}/unet")
    pipe.vae.save_pretrained(f"{save_path}/vae")
    
    

    print(f"Saved pipline modules at {save_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--text_encoder", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--sd_version",type=str, default="sd-14")
    parser.add_argument("--save_path", type=str, default="models/")
    

    args = parser.parse_args()
    save_path = os.path.join(args.save_path, args.sd_version)

    if args.pipeline is not None:
        pipeline_save(args.pipeline, save_path)
        
    else:
        # text encoder
        if args.text_encoder is not None:
            text_encoder_save(args.text_encoder, save_path)
        
        
