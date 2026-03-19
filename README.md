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

The top-level `data/` directory is currently unused and not tracked.
