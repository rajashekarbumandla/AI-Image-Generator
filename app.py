# app.py

import os
import time
from io import BytesIO

import streamlit as st
from PIL import Image

from models import load_sd_model
from generator import generate_images
from config import allowed

st.set_page_config(page_title="AI Image Generator", page_icon="🎨", layout="wide")


def load_image(path: str) -> Image.Image:
    return Image.open(path)


@st.cache_resource
def get_pipeline():
    return load_sd_model()


# Load pipeline once
pipe, device = get_pipeline()

st.title("🎨 AI-Powered Text-to-Image Generator")
st.write("Generate high-quality images from text prompts using an open-source Stable Diffusion model.")

with st.sidebar:
    st.header("Generation Settings")
    style = st.selectbox("Style preset", ["Photorealistic", "Artistic", "Cartoon"])
    num_images = st.slider("Number of images", 1, 4, 1)
    steps = st.slider("Inference steps", 10, 50, 30)
    guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5)

    st.markdown("**Hardware info**")
    st.write(f"Device: `{device}`")
    st.caption("GPU is recommended for faster generation. CPU will be slower but still works.")

prompt = st.text_area(
    "Enter your prompt",
    placeholder="e.g., a futuristic city at sunset, cinematic lighting"
)

negative_prompt = st.text_input(
    "Negative prompt (optional)",
    placeholder="e.g., low quality, blurry, distorted, nsfw"
)

generate_btn = st.button("Generate Images")

if generate_btn:
    if not prompt.strip():
        st.error("Please enter a valid prompt.")
    elif not allowed(prompt):
        st.error("Prompt contains blocked content. Please use a safe, appropriate prompt.")
    else:
        with st.spinner("Generating images... this may take a while on CPU"):
            start_time = time.time()
            metadata_list = generate_images(
                pipe=pipe,
                device=device,
                user_prompt=prompt.strip(),
                style=style,
                negative_prompt=negative_prompt.strip() if negative_prompt else None,
                num_images=num_images,
                num_inference_steps=steps,
                guidance_scale=guidance,
                output_dir="outputs"
            )
            total_time = time.time() - start_time

        st.success(f"Generation completed in ~{total_time:.1f} seconds.")
        st.write("### Results")

        cols = st.columns(num_images)
        for i, meta in enumerate(metadata_list):
            img_path = meta["paths"]["png"]
            img = load_image(img_path)

            with cols[i]:
                st.image(img, caption=f"Image {i+1}")
                # Prepare download buttons
                buf_png = BytesIO()
                img.save(buf_png, format="PNG")
                buf_png.seek(0)

                buf_jpg = BytesIO()
                img.convert("RGB").save(buf_jpg, format="JPEG", quality=95)
                buf_jpg.seek(0)

                st.download_button(
                    label="Download PNG",
                    data=buf_png,
                    file_name=os.path.basename(meta["paths"]["png"]),
                    mime="image/png",
                    key=f"png_{i}"
                )
                st.download_button(
                    label="Download JPG",
                    data=buf_jpg,
                    file_name=os.path.basename(meta["paths"]["jpg"]),
                    mime="image/jpeg",
                    key=f"jpg_{i}"
                )

        with st.expander("Generation metadata"):
            st.json(metadata_list)

st.markdown("---")
st.caption("Please use this tool responsibly. Do not generate harmful, illegal, or explicit content.")
