import os
import requests
import json
import base64
from diffusers import ( DiffusionPipeline,
StableDiffusionImg2ImgPipeline,
StableDiffusionInpaintPipeline,
EulerAncestralDiscreteScheduler)
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from accelerate import PartialState

from fastapi import FastAPI, Response, Request

from helper_functions import weight_keyword, create_prompt, image_grid, choose_scheduler, load_pipelines, models  

app = FastAPI()

txt2img_model, img2img_model, inpaint_model, image_variation_model = load_pipelines(model_id = "SG161222/Realistic_Vision_V1.4", scheduler=EulerAncestralDiscreteScheduler) #revision="fp16", #torch_dtype=torch.float16)

@app.get("/")
def hello_world():
    return "Welcome to my REST API"

@app.post("/txt2img")
async def txt2img(request : Request):
    imagesstr = []
    data = await request.json()
    prompt = data["prompt"]
    steps = data["steps"]
    cfg = data["guidance scale"]
    num_images = data["number images"]
    images = txt2img_model(prompt, num_inference_steps=steps, guidance_scale=cfg, num_images_per_prompt=num_images).images
    for num_image in range(len(images)):
        image = images[num_image]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        imgstr = base64.b64encode(buffer.getvalue())
        imagesstr.append(imgstr)
    grid = image_grid(images)
    return Response(content=imagesstr, media_type="image/png")

@app.route("/inpaint", methods=["POST"])
def inpaint(prompt : str , path : str, steps : int, cfg : float, num_images : int, input_image, mask_image):
    imagesstr = []
    images = inpaint_model(prompt, image=input_image, mask_image=mask_image, num_inference_steps=steps, guidance_scale=cfg, num_images_per_prompt=num_images).images
    for num_image in range(len(images)):
        image = images[num_image]
        image.save(f"{path}/Test Image {num_image}.png")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        imgstr = base64.b64encode(buffer.getvalue())
        imagesstr.append(imgstr)
    grid = image_grid(images)
    grid.save(f"{path}/Image Grid.png")
    return Response(content=imagesstr, media_type="image/png")