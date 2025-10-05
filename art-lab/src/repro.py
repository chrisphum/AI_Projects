import yaml, pathlib
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from src.determinism import make_deterministic
from src.logging_utils import append_ledger, git_commit_sha

def load_sampler(pipeline, name:str):
    if name.lower() in {"euler_a","euler-ancestral"}:
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    else:
        raise ValueError(f"Unsupported sampler: {name}")
    return pipeline

def make_grid(images, rows, cols):
    assert len(images) == rows*cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols*w, rows*h))
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(im, (c*w, r*h))
    return grid

def main(cfg_path="configs/baseline.yaml", out="viz/baseline_grid.png"):
    cfg = yaml.safe_load(open(cfg_path))
    # Deterministic 
    make_deterministic(cfg["seed"])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model"], torch_dtype=torch.float16 if device=="mps" else torch.float32
    )
    pipe = load_sampler(pipe, cfg["sampler"])
    pipe = pipe.to(device)

    variants = cfg["grid"]["variants"]
    imgs = []
    for v in variants:
        prompt = v.get("prompt", cfg["prompt"])
        neg = v.get("negative_prompt", cfg.get("negative_prompt",""))
        steps = v.get("steps", cfg["num_inference_steps"])
        gscale = v.get("cfg", cfg["guidance_scale"])
        width = v.get("width", cfg["width"])
        height = v.get("height", cfg["height"])

        # Deterministic 
        gen = torch.Generator(device=device).manual_seed(cfg["seed"])
        result = pipe(
            prompt, negative_prompt=neg,
            guidance_scale=gscale, num_inference_steps=steps,
            width=width, height=height, generator=gen
        )
        imgs.append(result.images[0])

    grid = make_grid(imgs, cfg["grid"]["rows"], cfg["grid"]["cols"])
    pathlib.Path("viz").mkdir(parents=True, exist_ok=True)
    grid.save(out)

    append_ledger({
        "seed": cfg["seed"],
        "cfg": cfg["guidance_scale"],
        "steps": cfg["num_inference_steps"],
        "sampler": cfg["sampler"],
        "lora_scales": "",  # none for baseline
        "commit": git_commit_sha(),
        "config_path": cfg_path,
        "notes": "baseline grid",
        "grid_path": out
    })

if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/baseline.yaml"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "viz/baseline_grid.png"
    main(cfg_path, out_path)