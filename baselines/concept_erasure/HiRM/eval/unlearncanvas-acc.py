"""
refernce : https://github.com/OPTML-Group/UnlearnCanvas/tree/master/machine_unlearning/evaluation/sampling_unlearned_models

"""

import argparse
import timm
from torchvision import transforms
import torch
torch.hub.set_dir("cache")
import sys
from PIL import Image
from tqdm import tqdm
import os
sys.path.append("")
# from constants.const import class_available, theme_available

 

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_concept", type=str, default=None)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--style_ckpt", type=str, required=True)
    parser.add_argument("--object_ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=[188])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

   
    theme_available = ["Abstractionism", "Artist_Sketch", "Blossom_Season", "Bricks", "Byzantine", "Cartoon",
                       "Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Cubism", "Dadaism", "Dapple",
                       "Defoliation", "Early_Autumn", "Expressionism", "Fauvism", "French", "Glowing_Sunset",
                       "Gorgeous_Love", "Greenfield", "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams",
                       "Magic_Cube", "Meta_Physics", "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire",
                       "Pastel", "Pencil_Drawing", "Picasso", "Pop_Art", "Red_Blue_Ink", "Rust", "Seed_Images",
                       "Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Ukiyoe",
                       "Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter"]

    class_available = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers",
                         "Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers", "Trees", "Waterfalls"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set input and output directories (use subfolder if theme is specified)
    input_dir = os.path.join(args.input_dir, args.target_concept) if args.target_concept is not None else args.input_dir
    save_path = os.path.join(args.save_path, args.target_concept) if args.target_concept is not None else args.output_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set the result file paths for style and class separately
    if args.target_concept is not None:
        style_output_path = os.path.join(save_path, f"{args.target_concept}.pth")
        class_output_path = os.path.join(save_path, f"{args.target_concept}-obj.pth")
    else:
        style_output_path = os.path.join(save_path, "result_style.pth")
        class_output_path = os.path.join(save_path, "result_class.pth")

    # 1. Create style model and load checkpoint
    style_model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True).to(device)
    num_style_classes = len(theme_available)
    style_model.head = torch.nn.Linear(1024, num_style_classes).to(device)
    style_ckpt = args.style_ckpt
    style_model.load_state_dict(torch.load(style_ckpt, map_location=device)["model_state_dict"])
    style_model.eval()

    # 2. Create class model and load checkpoint
    class_model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True).to(device)
    num_class = len(class_available)
    class_model.head = torch.nn.Linear(1024, num_class).to(device)
    class_ckpt = args.object_ckpt
    class_model.load_state_dict(torch.load(class_ckpt, map_location=device)["model_state_dict"])
    class_model.eval()

    # Initialize dictionaries for saving results
    style_results = {
        "test_theme": args.target_concept if args.target_concept is not None else "sd",
        "input_dir": args.input_dir,
        "loss": {theme: 0.0 for theme in theme_available},
        "acc": {theme: 0.0 for theme in theme_available},
        "pred_loss": {theme: 0.0 for theme in theme_available},
        "misclassified": {theme: {other_theme: 0 for other_theme in theme_available} for theme in theme_available},
    }

    class_results = {
        "test_theme": args.target_concept if args.target_concept is not None else "sd",
        "input_dir": args.input_dir,
        "loss": {cls: 0.0 for cls in class_available},
        "acc": {cls: 0.0 for cls in class_available},
        "pred_loss": {cls: 0.0 for cls in class_available},
        "misclassified": {cls: {other_cls: 0 for other_cls in class_available} for cls in class_available},
    }

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Evaluate style images for each theme in theme_available
    for idx, test_theme in tqdm(enumerate(theme_available), desc="Style Evaluation"):
        theme_label = idx 
        for seed in args.seed:
            for object_class in class_available:
                img_path = os.path.join(input_dir, f"{test_theme}_{object_class}_seed{seed}.jpg")
                image = Image.open(img_path)
                target_image = image_transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    res = style_model(target_image)
                    label = torch.tensor([theme_label]).to(device)
                    loss = torch.nn.functional.cross_entropy(res, label)
                    res_softmax = torch.nn.functional.softmax(res, dim=1)
                    pred_loss = res_softmax[0][theme_label]
                    pred_label = torch.argmax(res)
                    pred_success = (torch.argmax(res) == theme_label).sum()
                style_results["loss"][test_theme] += loss.item()
                style_results["pred_loss"][test_theme] += pred_loss.item()
                style_results["acc"][test_theme] += (pred_success.item() * 1.0 / (len(class_available) * len(args.seed)))
                misclassified_as = theme_available[pred_label.item()]
                style_results["misclassified"][test_theme][misclassified_as] += 1

        if not args.dry_run:
            torch.save(style_results, style_output_path)

    # For each theme, evaluate images for each class
    for test_theme in tqdm(theme_available, desc="Class Evaluation"):
        for seed in args.seed:
            for idx, object_class in enumerate(class_available):
                theme_label = idx  
                img_path = os.path.join(input_dir, f"{test_theme}_{object_class}_seed{seed}.jpg")
                image = Image.open(img_path)
                target_image = image_transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    res = class_model(target_image)
                    label = torch.tensor([theme_label]).to(device)
                    loss = torch.nn.functional.cross_entropy(res, label)
                    res_softmax = torch.nn.functional.softmax(res, dim=1)
                    pred_loss = res_softmax[0][theme_label]
                    pred_success = (torch.argmax(res) == theme_label).sum()
                    pred_label = torch.argmax(res)
                class_results["loss"][object_class] += loss.item()
                class_results["pred_loss"][object_class] += pred_loss.item()
                class_results["acc"][object_class] += (pred_success.item() * 1.0 / (len(theme_available) * len(args.seed)))
                misclassified_as = class_available[pred_label.item()]
                class_results["misclassified"][object_class][misclassified_as] += 1

        if not args.dry_run:
            torch.save(class_results, class_output_path)
