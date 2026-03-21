#!/usr/bin/env python3
"""
Generate images from compositional intent prompts using open-source T2I models.

Usage:
    python gen_images.py                           # default: SD3.5-medium, all prompts
    python gen_images.py --model sd1.5             # SD v1.5 (no safety checker)
    python gen_images.py --model sd1.5-safe        # SD v1.5 + safety checker enabled
    python gen_images.py --model sd3               # use SD3
    python gen_images.py --model sdxl              # use SDXL
    python gen_images.py --prompt_ids 0 1 2        # specific prompt IDs only
    python gen_images.py --seeds 42 123 456 789    # custom seeds
"""

import json
import argparse
import time
from pathlib import Path

import torch
from diffusers import (
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
)

# ── paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = SCRIPT_DIR / "gen_prompts_v1.json"
OUTPUT_DIR = SCRIPT_DIR / "generated_images"

# ── model configs ──────────────────────────────────────────────────
MODEL_CONFIGS = {
    "sd1.4": {
        "model_id": "CompVis/stable-diffusion-v1-4",
        "pipeline_cls": StableDiffusionPipeline,
        "default_steps": 50,
        "default_guidance": 7.5,
        "resolution": (512, 512),
        "safety_checker": False,
    },
    "sd1.5": {
        "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "pipeline_cls": StableDiffusionPipeline,
        "default_steps": 50,
        "default_guidance": 7.5,
        "resolution": (512, 512),
        "safety_checker": False,
    },
    "sd1.5-safe": {
        "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "pipeline_cls": StableDiffusionPipeline,
        "default_steps": 50,
        "default_guidance": 7.5,
        "resolution": (512, 512),
        "safety_checker": True,
    },
    "sd3.5": {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "pipeline_cls": StableDiffusion3Pipeline,
        "default_steps": 28,
        "default_guidance": 7.0,
        "resolution": (1024, 1024),
    },
    "sd3": {
        "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "pipeline_cls": StableDiffusion3Pipeline,
        "default_steps": 28,
        "default_guidance": 7.0,
        "resolution": (1024, 1024),
    },
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_cls": StableDiffusionXLPipeline,
        "default_steps": 30,
        "default_guidance": 7.5,
        "resolution": (1024, 1024),
    },
}

DEFAULT_SEEDS = [42, 123, 456, 789]


def load_pipeline(model_key: str):
    cfg = MODEL_CONFIGS[model_key]
    print(f"Loading {cfg['model_id']}...")
    kwargs = {"torch_dtype": torch.float16}
    # disable safety checker unless explicitly enabled
    if not cfg.get("safety_checker", False):
        kwargs["safety_checker"] = None
    pipe = cfg["pipeline_cls"].from_pretrained(cfg["model_id"], **kwargs)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe, cfg


def generate(pipe, cfg, prompt: str, seed: int):
    """Returns (image, nsfw_flagged). image is None if blocked by safety checker."""
    generator = torch.Generator("cpu").manual_seed(seed)
    w, h = cfg["resolution"]
    output = pipe(
        prompt=prompt,
        num_inference_steps=cfg["default_steps"],
        guidance_scale=cfg["default_guidance"],
        width=w,
        height=h,
        generator=generator,
    )
    nsfw_flagged = False
    if hasattr(output, "nsfw_content_detected") and output.nsfw_content_detected:
        nsfw_flagged = output.nsfw_content_detected[0]
    return output.images[0], nsfw_flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sd3.5", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    parser.add_argument("--prompt_ids", type=int, nargs="*", default=None,
                        help="Generate only these prompt IDs")
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    with open(args.prompts) as f:
        all_prompts = json.load(f)

    if args.prompt_ids is not None:
        prompts = [p for p in all_prompts if p["id"] in args.prompt_ids]
    else:
        prompts = all_prompts

    print(f"Model: {args.model} | Prompts: {len(prompts)} | Seeds: {args.seeds}")
    print(f"Total images to generate: {len(prompts) * len(args.seeds)}")

    pipe, cfg = load_pipeline(args.model)

    model_dir = args.output / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []
    n_flagged = 0
    for p in prompts:
        pid = p["id"]
        cat = p["category"]
        prompt_text = p["prompt"]

        for seed in args.seeds:
            fname = f"{pid:03d}_{cat}_seed{seed}.png"
            fpath = model_dir / fname

            if fpath.exists():
                print(f"  [skip] {fname}")
                continue

            t0 = time.time()
            image, nsfw_flagged = generate(pipe, cfg, prompt_text, seed)
            elapsed = time.time() - t0

            image.save(fpath)
            flag_str = " [NSFW FLAGGED]" if nsfw_flagged else ""
            print(f"  [done] {fname} ({elapsed:.1f}s){flag_str}")
            if nsfw_flagged:
                n_flagged += 1

            results.append({
                "id": pid,
                "category": cat,
                "prompt": prompt_text,
                "seed": seed,
                "model": args.model,
                "file": str(fpath.relative_to(args.output)),
                "nsfw_flagged": nsfw_flagged,
            })

    # save generation log
    log_path = model_dir / "generation_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    total = len(results)
    print(f"\nDone. {total} images saved to {model_dir}")
    if n_flagged > 0:
        print(f"Safety checker flagged: {n_flagged}/{total} ({n_flagged/total*100:.1f}%)")
    elif cfg.get("safety_checker", False):
        print(f"Safety checker flagged: 0/{total} (0.0%) — compositional prompts bypassed filter")


if __name__ == "__main__":
    main()
