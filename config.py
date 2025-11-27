# config.py

DEFAULT_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "outputs"

BANNED = ["nude", "sex", "porn", "nsfw", "kill", "blood", "murder"]

def allowed(prompt: str) -> bool:
    p = prompt.lower()
    return not any(bad in p for bad in BANNED)
