"""
refernce : https://github.com/OPTML-Group/UnlearnCanvas/tree/master/machine_unlearning/evaluation/sampling_unlearned_models

"""


from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel,StableDiffusionPipeline
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import argparse
import os
import sys
from diffusers import AutoencoderKL, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
sys.path.append("..")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generateImages',
        description='Generate Images using Diffusers Code')
    parser.add_argument('--target_concept', type=str, required=True)
    parser.add_argument('--ckpt', help='path to the text-encoder ckpt', type=str, required=True)
    parser.add_argument('--pipe_dir', help='path to the saved SD1.5 fine‑tuned on the UnlearnCanvas pipeline')
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--seed', help='seed', type=int, required=False, default=188)
    parser.add_argument('--steps', help='ddim steps of inference used to train', type=int, required=False, default=100)

    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    save_path = os.path.join(args.save_path, args.target_concept)
    os.makedirs(save_path, exist_ok=True)
    
    theme_available=["Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine", "Cartoon",
                    "Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism", "Dadaism", "Dapple",
                    "Defoliation", "Early_Autumn", "Expressionism", "Fauvism", "French", "Glowing_Sunset",
                    "Gorgeous_Love", "Greenfield", "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams",
                    "Magic_Cube", "Meta_Physics", "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire",
                    "Pastel", "Pencil_Drawing", "Picasso", "Pop_Art", "Red_Blue_Ink", "Rust","Seed_Images",
                    "Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Ukiyoe",
                    "Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter"]
    
    
    class_available = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers",
                        "Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers", "Trees", "Waterfalls"]
                        
    
              
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    
    text_encoder = CLIPTextModel.from_pretrained(args.ckpt)
    vae = AutoencoderKL.from_pretrained(os.path.join(args.pipe_dir,'vae'))
    unet = UNet2DConditionModel.from_pretrained(os.path.join(args.pipe_dir,'unet-ema'))
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(args.pipe_dir,'tokenizer'))
    

    scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1
    )

    remover_model = StableDiffusionPipeline(
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    
    remover_model = remover_model.to(args.gpu)

    

    for test_theme in theme_available:
        for object_class in class_available:
            output_path = f"{save_path}/{test_theme}_{object_class}_seed{args.seed}.jpg"
            if os.path.exists(output_path):
                print(f"Detected! Skipping {output_path}")
                continue
            prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."
            print(prompt)

            generator = torch.manual_seed(args.seed)
            
            removal_image = remover_model(prompt,num_inference_steps=100,generator=generator).images[0]
            removal_image.save(output_path)

