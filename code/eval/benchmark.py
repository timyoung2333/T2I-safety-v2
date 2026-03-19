import time, torch, pandas as pd
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusion3Pipeline
import sys

model_type = sys.argv[1] if len(sys.argv) > 1 else 'sd1.4'

df = pd.read_csv('/scratch/gilbreth/yang1683/projects/vlm_safety/T2I-safety-v2/code/eval/data/mscoco.csv')
prompts = list(df['text'])[:10]

if model_type == 'sd1.4':
    pipe = DiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', safety_checker=None, torch_dtype=torch.float16).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    steps = 20
elif model_type == 'sd3.5-medium':
    pipe = StableDiffusion3Pipeline.from_pretrained('stabilityai/stable-diffusion-3.5-medium', torch_dtype=torch.float16).to('cuda')
    steps = 28

generator = torch.Generator('cuda').manual_seed(0)
start = time.time()
for i, p in enumerate(prompts):
    img = pipe(p, num_inference_steps=steps, guidance_scale=7.5, generator=generator).images[0]
    print(f'[{model_type}] {i+1}/10 done')
elapsed = time.time() - start
print(f'{model_type}: {elapsed:.1f}s for 10 images, {elapsed/10:.2f}s/image, estimated 1000: {elapsed/10*1000/60:.1f} min')
