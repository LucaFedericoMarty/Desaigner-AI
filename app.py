import streamlit as st
from diffusers import ( DiffusionPipeline,
StableDiffusionImg2ImgPipeline,
StableDiffusionInpaintPipeline,
EulerAncestralDiscreteScheduler)
import torch
from PIL import Image

from helper_functions import create_prompt, image_grid, choose_scheduler, load_pipelines, models

txt2img_model, img2img_model, inpaint_model, imgvariation_model = load_pipelines(model_id = "SG161222/Realistic_Vision_V1.4", 
                                                            #revision="fp16", 
                                                            #torch_dtype=torch.float16, 
                                                            scheduler=EulerAncestralDiscreteScheduler)

def txt2img(prompt : str , steps : int, cfg : float, num_images : int):
    images = txt2img_model(prompt, num_inference_steps=steps, guidance_scale=cfg, num_images_per_prompt=num_images).images
    grid = image_grid(images)
    return images

def imgvariation(image : Image , steps : int, cfg : float, num_images : int):
    images = imgvariation_model(image=image, num_inference_steps=steps, guidance_scale=cfg, num_images_per_prompt=num_images).images
    grid = image_grid(images)
    return images

st.title("DesAIgner App Deployment for testing :house_with_garden:")

st.markdown("## Inputs for creating an image ##")

budget = st.selectbox("Budget", ('Low', 'Medium', 'High'))

style = st.selectbox("Style", ('Minimalist', 'Modern', 'Eclectic', 'Traditional', 'Bohemian', 'Shabby Chic', 'Coastal', 'Rustic', 'Industrial', 'Scandinavian', 'Mediterranean', 'Art Deco', 'Transitional', 'Farmhouse', 'Oriental'))

environment = st.selectbox("Environment", ('Kitchen', 'Living Room', 'Bedroom', 'Bathroom'))

region_weather = st.selectbox("Region", ('Cold', 'Hot', 'Snowy', 'Rainy', 'Tropical'))

num_images = st.slider('How many photos do you want?', 1, 10, 4)

prompt = create_prompt(budget=budget, style=style, environment=environment, region_weather=region_weather)

if st.button("Create image :sparkles:"):
    images = txt2img(prompt=prompt, steps=50, cfg=4.5, num_images=num_images)
    cols = st.columns(spec=4, gap="medium")
    for num_image in range(len(images)):
        image = images[num_image]
        if st.button (cols[num_images].image(image=image)):
            images_variation = imgvariation(image=image, steps=50, cfg=4.5, num_images=num_images)
            for num_image in range(len(images_variation)):
                image_variation = images_variation[num_image]
                cols[num_images].image(image=image_variation)



