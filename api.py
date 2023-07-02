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
from pydantic import BaseModel, ConfigDict
from typing import Optional

from helper_functions import weight_keyword, create_prompt, image_grid, choose_scheduler, load_pipelines, zipfiles, models  

app = FastAPI()

class design(BaseModel):
    #model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: Optional[str]
    steps: int
    guidance_scale : float
    num_images: int
    #input_image: Optional[Image.Image]

txt2img_model, img2img_model, inpaint_model, image_variation_model = load_pipelines(model_id = "SG161222/Realistic_Vision_V1.4", scheduler=EulerAncestralDiscreteScheduler) #revision="fp16", #torch_dtype=torch.float16)

@app.get("/")
def hello_world():
    return "Welcome to my REST API"

@app.post("/txt2img")
def txt2img(design_request : design):
    imagesstr = []
    prompt = design_request.prompt
    steps = design_request.steps
    cfg = design_request.guidance_scale
    num_images = design_request.num_images
    images = txt2img_model(prompt, num_inference_steps=steps, guidance_scale=cfg, num_images_per_prompt=num_images).images
    for num_image in range(len(images)):
        image = images[num_image]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        imgstr = base64.b64encode(buffer.getvalue())
        imagesstr.append(imgstr)
    grid = image_grid(images)
    return zipfiles(imagesstr)

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