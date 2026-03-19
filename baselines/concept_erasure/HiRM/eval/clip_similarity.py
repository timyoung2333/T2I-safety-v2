from PIL import Image
import argparse
import os, re
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


device = "cuda" if torch.cuda.is_available() else "cpu"


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_clip_similarity(image_folder, csv_path):

    if not os.path.exists(image_folder):
        print(f"Error: {image_folder} does not exist.")
        return

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return
    
    df = pd.read_csv(csv_path)

    if 'prompt' not in df.columns:
        print("Error: CSV file does not contain 'prompt' column.")
        return

    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    images = sorted_nicely(images)

    num_prompts = len(df)
    num_images = len(images)

    if not images:
        print("No images found in the folder.")
        return

    num_items = min(num_images, num_prompts)
    scores = []
    
    for idx, image_name in enumerate(tqdm(images, desc="Processing Images")):
        try:
            if idx >= len(df):
                print(f"Skipping {image_name}: No matching prompt found in CSV.")
                continue
            
            text_prompt = df.loc[idx, 'prompt']
            image_path = os.path.join(image_folder, image_name)
            image = Image.open(image_path).convert("RGB")

           
            text_inputs = processor.tokenizer([text_prompt], return_tensors="pt", padding=True).to(device)
            image_inputs = processor(images=image, return_tensors="pt").to(device)

          
            inputs = {**text_inputs, **image_inputs}

            with torch.no_grad():
                outputs = model(**inputs)
                clip_score = outputs.logits_per_image[0][0].item()
                scores.append(clip_score)
                
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            print(f"Prompt: {text_prompt}")  
    
    if scores:
        print(f"Mean CLIP score: {np.mean(scores):.4f}")
        
    else:
        print("No valid CLIP scores computed.")

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder_path", type=str, help='path to the generated images')
    parser.add_argument("--csv_file_path", type=str, help='path to the COCO_30k_10k')
    parser.add_argument("--seed", type=int, nargs="+", default=[188])
    
    args = parser.parse_args()


    
    compute_clip_similarity(args.image_folder_path, args.csv_file_path)
