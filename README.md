# T2I-safety-v2

Text-to-Image safety research project.

## Project Structure

```
├── baselines/       # Baseline methods (HiRM, SPEED, etc.)
├── code/            # Our code
│   ├── eval/        # Evaluation scripts
│   ├── benchmark/   # Benchmark code
│   └── method/      # Our method
├── data/            # Datasets (not tracked, see below)
├── refs/            # Reference papers
├── results/         # Experiment results (not tracked)
└── writing/         # Paper drafts and progress tracker
```

## Data Setup

We use the MS-COCO data provided by SPEED. The relevant prompts and datasets are located in:

- `baselines/concept_erasure/SPEED/data/` — MS-COCO prompts (`mscoco.csv`), I2P benchmark, celebrity lists, etc.
- `baselines/concept_erasure/HiRM/datasets/` — COCO 30k/10k splits, I2P, MMA-Diffusion prompts

The top-level `data/` directory contains MS-COCO val2014 images (40,504 images).

## Evaluation: COCO Generation & CLIP Score

### Generate images from COCO captions

```bash
# SD 1.4 (20 steps, guidance 7.5)
python code/eval/coco_sampling.py --model_type sd1.4 --contents coco --mode original

# SD 3.5-Medium (28 steps, guidance 7.5)
python code/eval/coco_sampling.py --model_type sd3.5-medium --contents coco --mode original
```

Generated images are saved to `results/coco_eval/<model>/coco/original/`.

### Calculate CLIP Score

```bash
python code/eval/clip_score_cal.py --contents coco --root_path results/coco_eval/<model>
```

Uses `openai/clip-vit-large-patch14`. Scores are reported as CS × 100.
