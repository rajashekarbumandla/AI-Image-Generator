# prompt_utils.py

STYLE_PRESETS = {
    "Photorealistic": "photorealistic, 8k uhd, ultra detailed, high dynamic range, professional photography",
    "Artistic": "digital art, concept art, matte painting, highly detailed, artstation trending",
    "Cartoon": "cartoon style, 2d illustration, clean lines, vibrant colors, highly detailed"
}

BASE_QUALITY_TAGS = "highly detailed, sharp focus, 4k, ultra quality"

def build_prompt(user_prompt: str, style: str) -> str:
    style_tags = STYLE_PRESETS.get(style, "")
    full_prompt = f"{user_prompt}, {BASE_QUALITY_TAGS}"
    if style_tags:
        full_prompt += f", {style_tags}"
    return full_prompt

def build_negative_prompt(user_negative: str | None) -> str:
    base_negative = "low quality, blurry, distorted, extra limbs, deformed, watermark, text, nsfw"
    if user_negative:
        return base_negative + ", " + user_negative
    return base_negative
