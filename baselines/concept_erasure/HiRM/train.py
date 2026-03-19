"""
refernce : 
https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
https://github.com/centerforaisafety/wmdp/blob/main/rmu/unlearn.py
https://github.com/fmp453/few-shot-erasing/blob/main/train.py
"""

import numpy as np
import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
from collections import OrderedDict
import cv2
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
import utils
from transformers import CLIPTextModel, CLIPTokenizer
import random
from diffusers import DDIMScheduler

class Configuration:
    def __init__(self, *args, **kwargs) -> None:
        args = args[0]
        self.target_concept = args["target_concept"]
        self.concept_type = args["concept_type"]
        
        self.optimizer_name = args["optim"]
        self.lr = args["lr"]
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.eps = args["eps"]
        self.weight_decay = args["weight_decay"]

        self.gpu_id = args["gpu_id"]
        
        self.save_path = args["save_path"]
        self.num_epoch = args["epochs"]
        self.clip_version = args["clip_ver"]
        self.sd_version = args["sd_ver"]
        self.text_encoder_path = args["text_encoder_path"]
        self.tokenizer_version = args["tokenizer_version"]
        self.seed = args["seed"]
        self.coeff = args['steering_coeff']
        self.verbose = args['verbose']
       
        
        
       
def get_text_embeddings(text_encoder, tokenized_text):
    # ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L377

    device = text_encoder.device
    weight_dtype = text_encoder.dtype

    text_embedding = text_encoder(tokenized_text.to(device))[0].to(weight_dtype)
    return text_embedding


def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    
    
    
    def hook(module, input, output):
        
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]


def train(config: Configuration):
    
    target_concept = config.target_concept
    concept_type = config.concept_type

    seed = config.seed
    coeff = config.coeff
    save_path = os.path.join(config.save_path,target_concept)
    num_epochs = config.num_epoch
    device = f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"

    
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    tokenizer, text_encoder, scheduler = utils.load_models_from_local_optioned_path(
        text_encoder_path=config.text_encoder_path,
        tokenizer_version=config.tokenizer_version
    )
   
    
   
    
    
    text_encoder.to(device)
    
    if concept_type in ["style","object"]:
        lr = 5e-5
        
    else:
        lr = 1e-4
        

    print(lr)
    print(coeff)

    # optimizer setting
    optimizer = utils.get_optimizer(
        text_encoder.parameters(),
        config.optimizer_name,
        lr,
        (config.beta1, config.beta2),
        config.weight_decay,
        config.eps,
    )

    control_vectors_list = []
    

   
    torch.manual_seed(seed)#60 #4
    
    # set random unit vector 
    single_random_vector = torch.rand(1, 77, 768, dtype=torch.float32, device=device)
    random_vector = single_random_vector.repeat(4, 1, 1) 
    control_vec = random_vector[0] / random_vector[0].norm() * coeff
    control_vectors_list.append(control_vec)

    history = {"loss": []}

    
    
    cnt = 0
    pbar = trange(0, num_epochs, desc="Epoch")

    # set upate layer
    text_encoder = utils.freeze_and_unfreeze_text_encoder(text_encoder, method="first-attn")
    
    #set target layer
    target_model = text_encoder
    target_model.to(torch.float32)
    target_module = text_encoder.text_model.encoder.layers[11].self_attn.out_proj

    
    
    tokenized = tokenizer(
            target_concept,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
    
    start = time.perf_counter()

    for epoch in pbar:

        loss_avg = 0
       
        text_encoder.train()
        
        text_embedding = get_text_embeddings(
            text_encoder=text_encoder, 
            tokenized_text=tokenized
        )
        
        
        tokenized = tokenized.to(device)
        control_vec = torch.stack(control_vectors_list).to(device) 
        
        
        unlearn_inputs = {
                            "input_ids": tokenized,
                            
                        }

        #forget loss
        updated_forget_activations = forward_with_cache(
                target_model, unlearn_inputs, module=target_module, no_grad=False
            ).to(device)
        
        forget_loss = F.mse_loss(
                updated_forget_activations, control_vec
            )

        loss = forget_loss 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_avg += loss.detach().item()
        cnt += 1
        
        history["loss"].append(loss.detach().item())
        
        pbar.set_postfix(OrderedDict(loss=loss_avg / (cnt + 1e-9)))
        

    text_encoder.save_pretrained(f"{save_path}/epoch-{epoch+1}")
        
       
    end = time.perf_counter()

    print(f"Time : {end - start}")
    
    utils.plot_loss(history, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # concept to erase
    parser.add_argument("--target_concept", type=str, required=True, help="concept to erase. for example, 'Van Gogh'")
    parser.add_argument("--concept_type", type=str, required=True, choices=["style","object","nsfw"])
    #set seed
    parser.add_argument("--seed", type=int, default=42)
    
    # optimizer setting
    parser.add_argument("--optim", type=str, default="AdamW", choices=["Adam", "AdamW", "Adadelta", "Adagrad"])
    parser.add_argument("--lr", type=float, default=1e-4) #nsfw : 1e-4, ob/st : 5e-5
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    

    # steering coefficient
    parser.add_argument(
        "--steering_coeff",
        type=float,
        required=False,default=5000,
        help="Steer vector weight in order of topic",
    )
   

    # other training setting
    parser.add_argument("--epochs", type=int, default=50)

    # GPU ID
    parser.add_argument("--gpu_id", type=int, default=0)
    # save path
    parser.add_argument("--save_path", type=str, default=None)
    
    # model version setting
    parser.add_argument("--clip_ver", type=str, default="oa", choices=["oa", "oa-336"])
    parser.add_argument("--sd_ver", type=str, default="14", choices=["14", "15"])
    parser.add_argument("--text_encoder_path", type=str, default="models/sd-14/text_encoder")
    parser.add_argument("--tokenizer_version", type=str, default="openai/clip-vit-large-patch14")
    
    
    # Logging
    parser.add_argument("--verbose", action="store_true", help="Logging the activations norms and cosine at each step")
    
    args = vars(parser.parse_args())
    config = Configuration(args)
    
    train(config=config)
    
