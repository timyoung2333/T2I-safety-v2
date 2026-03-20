import warnings
warnings.filterwarnings("ignore")
import os, sys, pdb
import re
import copy
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from diffusers import StableDiffusion3Pipeline

from src.template import template_dict
from src.utils import seed_everything


@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument('--save_root', type=str, default='')
    parser.add_argument('--sd_ckpt', type=str, default='stabilityai/stable-diffusion-3.5-medium')
    parser.add_argument('--seed', type=int, default=0)
    # Sampling Config
    parser.add_argument('--mode', type=str, default='original', help='original, edit')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--total_timesteps', type=int, default=28)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--prompts', type=str, default=None)
    # Erasing Config
    parser.add_argument('--erase_type', type=str, default='', help='instance, style, celebrity')
    parser.add_argument('--target_concept', type=str, default='')
    parser.add_argument('--contents', type=str, default='')
    parser.add_argument('--edit_ckpt', type=str, default=None)
    args = parser.parse_args()
    assert args.num_samples >= args.batch_size

    bs = args.batch_size
    mode_list = args.mode.replace(' ', '').split(',')

    # region [If certain concept is already sampled, then skip it.]
    concept_list, concept_list_tmp = [], [item.strip() for item in args.contents.split(',')]
    if 'edit' in mode_list:
        for concept in concept_list_tmp:
            check_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept, 'edit')
            os.makedirs(check_path, exist_ok=True)
            if len(os.listdir(check_path)) != len(template_dict[args.erase_type]) * 10:
                concept_list.append(concept)
    else:
        concept_list = concept_list_tmp
    if len(concept_list) == 0: sys.exit()
    # endregion

    # region [Prepare Models]
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.sd_ckpt, torch_dtype=torch.float16
    ).to('cuda')

    if 'edit' in mode_list:
        pipe_edit = StableDiffusion3Pipeline.from_pretrained(
            args.sd_ckpt, torch_dtype=torch.float16
        ).to('cuda')
        edit_path = args.edit_ckpt or os.path.join("logs/checkpoints", sorted(os.listdir("logs/checkpoints"))[-1])
        # Load edited transformer weights (add_k_proj / add_v_proj)
        edit_dict = torch.load(edit_path, map_location='cpu')
        pipe_edit.transformer.load_state_dict(edit_dict, strict=False)
    # endregion

    # Sampling process
    seed_everything(args.seed, True)
    if args.prompts is None:
        prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]
    else:
        prompt_list = [[x.format(concept) for x in args.prompts.split(';')] for concept in concept_list]

    for i in range(int(args.num_samples // bs)):
        generator = torch.Generator('cuda').manual_seed(args.seed + i)
        for concept, prompts in zip(concept_list, prompt_list):
            for count, prompt in enumerate(prompts):

                save_images = {}

                if 'original' in mode_list:
                    images = pipe(
                        prompt=prompt,
                        num_inference_steps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        num_images_per_prompt=bs,
                        generator=generator,
                    ).images
                    save_images['original'] = images

                if 'edit' in mode_list:
                    images = pipe_edit(
                        prompt=prompt,
                        num_inference_steps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        num_images_per_prompt=bs,
                        generator=generator,
                    ).images
                    save_images['edit'] = images

                save_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept)
                for mode in mode_list: os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                if len(mode_list) > 1: os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)

                # Save images
                def combine_images_horizontally(imgs):
                    widths, heights = zip(*(img.size for img in imgs))
                    new_img = Image.new('RGB', (sum(widths), max(heights)))
                    for j, img in enumerate(imgs): new_img.paste(img, (sum(widths[:j]), 0))
                    return new_img

                for idx in range(len(save_images[mode_list[0]])):
                    save_filename = re.sub(r'[^\w\s]', '', prompt).replace(', ', '_') + f"_{int(idx + bs * i)}.png"
                    images_to_combine = []
                    for mode in mode_list:
                        save_images[mode][idx].save(os.path.join(save_path, mode, save_filename))
                        images_to_combine.append(save_images[mode][idx])
                    if len(mode_list) > 1:
                        img_combined = combine_images_horizontally(images_to_combine)
                        img_combined.save(os.path.join(save_path, 'combine', save_filename.replace('.png', '.jpg')))


if __name__ == '__main__':
    main()
