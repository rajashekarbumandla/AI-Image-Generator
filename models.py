# models.py

import torch
from diffusers import StableDiffusionPipeline
from config import DEFAULT_MODEL_NAME

def load_sd_model(model_name: str = DEFAULT_MODEL_NAME):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # we'll handle basic filtering manually
    )
    pipe.to(device)

    # Enable memory efficient attention if on GPU
    if device == "cuda":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe, device
