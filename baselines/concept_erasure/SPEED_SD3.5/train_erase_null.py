import os, re, pdb
import time
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from kmeans_pytorch import kmeans
from diffusers import StableDiffusion3Pipeline
from src.utils import seed_everything


def get_projected_embs(pipeline, prompts, device="cuda"):
    """Encode prompts through all three text encoders + context_embedder → (batch, seq, 1536)."""
    if isinstance(prompts, str):
        prompts = [prompts]
    prompt_embeds, _, _, _ = pipeline.encode_prompt(
        prompt=prompts, prompt_2=prompts, prompt_3=prompts
    )
    # Project 4096-dim combined embeddings to 1536-dim transformer hidden size
    projected = pipeline.transformer.context_embedder(prompt_embeds.to(device))
    return projected


def get_token_positions(prompts, tokenizer):
    """Get attention mask from CLIP-L tokenizer to find last subject token."""
    if isinstance(prompts, str):
        prompts = [prompts]
    tokens = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="pt")
    return tokens.attention_mask


def generate_perturbed_embs(ret_embs, P, erase_weight, num_per_sample, mini_batch=8):
    ret_embs = ret_embs.squeeze(1)
    out_embs, norm_list = [], []
    for i in range(0, ret_embs.size(0), mini_batch):
        mini_ret_embs = ret_embs[i:i + mini_batch]
        for _ in range(num_per_sample):
            noise = torch.randn_like(mini_ret_embs)
            perturbed_embs = mini_ret_embs + noise @ P
            out_embs.append(perturbed_embs)
            norm_list.append(torch.matmul(perturbed_embs, erase_weight.T).norm(dim=1))
    out_embs = torch.cat(out_embs, dim=0)
    norm_list = torch.cat(norm_list, dim=0)
    return out_embs[norm_list > norm_list.mean()].unsqueeze(1)


@torch.no_grad()
def edit_model(args, pipeline, target_concepts, anchor_concepts, retain_texts,
               baseline=None, chunk_size=128, emb_size=1536, device="cuda"):
    """
    SD 3.5 adaptation of SPEED.

    Architecture mapping from SD 1.4:
      SD 1.4:  text (768) → attn2.to_k/to_v
      SD 3.5:  text (4096) → context_embedder (→1536) → attn.add_k_proj / attn.add_v_proj

    We operate in the 1536-dim space (after context_embedder projection).
    """
    I = torch.eye(emb_size, device=device)

    # Select layers to edit: add_k_proj / add_v_proj in joint attention blocks (weights only, skip bias)
    if args.params == 'KV':
        edit_dict = {k: v for k, v in pipeline.transformer.state_dict().items()
                     if ('attn.add_k_proj.weight' in k or 'attn.add_v_proj.weight' in k)}
    elif args.params == 'V':
        edit_dict = {k: v for k, v in pipeline.transformer.state_dict().items()
                     if 'attn.add_v_proj.weight' in k}
    elif args.params == 'K':
        edit_dict = {k: v for k, v in pipeline.transformer.state_dict().items()
                     if 'attn.add_k_proj.weight' in k}

    if baseline in ['SPEED']:
        # Null embedding: encode empty string, project, then cluster
        null_projected = get_projected_embs(pipeline, '', device)  # (1, seq, 1536)
        null_hidden = null_projected[0]  # (seq, 1536)
        # Use first 77 tokens (CLIP portion) for clustering, skip CLS token
        null_clip = null_hidden[1:77]
        cluster_ids, cluster_centers = kmeans(X=null_clip, num_clusters=3, distance='euclidean', device='cuda')
        K2 = torch.cat([null_hidden[[0], :], cluster_centers.to(device)], dim=0).T  # (1536, 4)
        I2 = torch.eye(len(K2.T), device=device)
    else:
        raise ValueError("Invalid baseline")

    # region [Target and Anchor]
    sum_anchor_target, sum_target_target = [], []
    for i in range(len(target_concepts)):
        # Get projected embeddings
        target_proj = get_projected_embs(pipeline, target_concepts[i], device)[0]  # (seq, 1536)
        anchor_proj = get_projected_embs(pipeline, anchor_concepts[i], device)[0]

        if target_concepts == ['nudity']:
            # Use all CLIP tokens (skip CLS)
            target_embs = target_proj[1:77, :]
            anchor_embs = anchor_proj[1:77, :]
        else:
            # Last subject token from CLIP-L portion (first 77 tokens)
            attn_mask = get_token_positions(target_concepts[i], pipeline.tokenizer)
            last_idx = attn_mask[0].sum().item() - 2  # last subject token (before EOS)
            target_embs = target_proj[[last_idx], :]

            attn_mask_a = get_token_positions(anchor_concepts[i], pipeline.tokenizer)
            last_idx_a = attn_mask_a[0].sum().item() - 2
            anchor_embs = anchor_proj[[last_idx_a], :]

        sum_target_target.append(target_embs.T @ target_embs)
        sum_anchor_target.append(anchor_embs.T @ target_embs)
    sum_target_target = torch.stack(sum_target_target).mean(0)
    sum_anchor_target = torch.stack(sum_anchor_target).mean(0)
    # endregion

    # region [Retain]
    last_ret_embs = []
    retain_texts = [text for text in retain_texts if not any(
        re.search(r'\b' + re.escape(concept.lower()) + r'\b', text.lower())
        for concept in target_concepts)]
    assert len(retain_texts) + len(target_concepts) == len(set(retain_texts + target_concepts))

    for j in range(0, len(retain_texts), chunk_size):
        chunk_texts = retain_texts[j:j + chunk_size]
        ret_proj = get_projected_embs(pipeline, chunk_texts, device)  # (batch, seq, 1536)

        if retain_texts == ['']:
            # Use all CLIP tokens (skip CLS)
            last_ret_embs.append(ret_proj[:, 1:77, :].permute(1, 0, 2))
        else:
            attn_masks = get_token_positions(chunk_texts, pipeline.tokenizer)
            last_subject_indices = attn_masks.sum(1) - 2
            last_ret_embs.append(
                ret_proj[torch.arange(ret_proj.size(0)), last_subject_indices].unsqueeze(1))

    last_ret_embs = torch.cat(last_ret_embs)
    last_ret_embs = last_ret_embs[torch.randperm(last_ret_embs.size(0))]  # shuffle
    # endregion

    for (layer_name, layer_weight) in tqdm(edit_dict.items(), desc="Model Editing"):

        erase_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ (I + sum_target_target).inverse()
        (U0, S0, V0) = torch.svd(layer_weight)
        P0_min = V0[:, -1:] @ V0[:, -1:].T

        if args.aug_num > 0 and not args.disable_filter:
            weight_norm_init = torch.matmul(last_ret_embs.squeeze(1), erase_weight.T).norm(dim=1)
            layer_ret_embs = last_ret_embs[weight_norm_init > weight_norm_init.mean()]
        else:
            layer_ret_embs = last_ret_embs

        sum_ret_ret, valid_num = [], 0
        for j in range(0, len(layer_ret_embs), chunk_size):
            chunk_ret_embs = layer_ret_embs[j:j + chunk_size]
            if args.aug_num > 0:
                chunk_ret_embs = torch.cat(
                    [chunk_ret_embs, generate_perturbed_embs(chunk_ret_embs, P0_min, erase_weight, num_per_sample=args.aug_num)], dim=0
                )
            valid_num += chunk_ret_embs.shape[0]
            sum_ret_ret.append((chunk_ret_embs.transpose(1, 2) @ chunk_ret_embs).sum(0))
        sum_ret_ret = torch.stack(sum_ret_ret, dim=0).sum(0) / valid_num

        if baseline == 'SPEED':
            U, S, V = torch.svd(sum_ret_ret)
            P = U[:, S < args.threshold] @ U[:, S < args.threshold].T
            M = (sum_target_target @ P + args.retain_scale * I).inverse()
            delta_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ P @ \
                (I - M @ K2 @ (K2.T @ P @ M @ K2 + args.lamb * I2).inverse() @ K2.T @ P) @ M

        edit_dict[layer_name] = layer_weight + args.edit_scale * delta_weight

    print(f"Current model status: Edited {str(target_concepts)} into {str(anchor_concepts)}")
    return edit_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument('--sd_ckpt', type=str, default='stabilityai/stable-diffusion-3.5-medium')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    # Erase Config
    parser.add_argument('--target_concepts', type=str, required=True)
    parser.add_argument('--anchor_concepts', type=str, required=True)
    parser.add_argument('--retain_path', type=str, default=None)
    parser.add_argument('--heads', type=str, default=None)
    parser.add_argument('--baseline', type=str, default='SPEED')
    # Hyperparameters
    parser.add_argument('--params', type=str, default='V')
    parser.add_argument('--aug_num', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=1e-1)
    parser.add_argument('--retain_scale', type=float, default=1.0)
    parser.add_argument('--lamb', type=float, default=0.0)
    parser.add_argument('--edit_scale', type=float, default=0.1, help='Scale factor for delta_weight (SD3.5 needs smaller values)')
    parser.add_argument('--disable_filter', action='store_true', default=False)
    args = parser.parse_args()
    device = torch.device("cuda")
    seed_everything(args.seed)

    target_concepts = [con.strip() for con in args.target_concepts.split(',')]
    anchor_concepts = args.anchor_concepts
    retain_path = args.retain_path

    file_suffix = "_".join(target_concepts[:5]) + f"_{len(target_concepts)}"
    anchor_concepts = [x.strip() for x in anchor_concepts.split(',')]
    if len(anchor_concepts) == 1:
        anchor_concepts = anchor_concepts * len(target_concepts)
        if anchor_concepts[0] == "":
            file_suffix += '-to_null'
        else:
            file_suffix += f'-to_{anchor_concepts[0]}'
    else:
        assert len(target_concepts) == len(anchor_concepts)
        file_suffix += f'-to_{anchor_concepts[0]}_etc'

    retain_texts = []
    if retain_path is not None:
        assert retain_path.endswith('.csv')
        df = pd.read_csv(retain_path)
        for head in args.heads.split(','):
            retain_texts += df[head.strip()].unique().tolist()
    else:
        retain_texts.append("")

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.sd_ckpt, torch_dtype=torch.float32
    ).to(device)

    edit_dict = edit_model(
        args=args,
        pipeline=pipeline,
        target_concepts=target_concepts,
        anchor_concepts=anchor_concepts,
        retain_texts=retain_texts,
        baseline=args.baseline,
        device=device,
    )

    save_path = args.save_path or "logs/checkpoints"
    file_name = args.file_name or f"{time.strftime('%Y%m%d-%H%M%S')}-sd3.5-{file_suffix}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(edit_dict, os.path.join(save_path, f"{file_name}.pt"))
