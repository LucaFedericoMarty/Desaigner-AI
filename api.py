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

from fastapi import FastAPI, Response

from helper_functions import weight_keyword, create_prompt, image_grid, choose_scheduler, load_pipeline, models  

app = FastAPI()

txt2img_model, img2img_model, inpaint_model = load_pipeline(model_id = "SG161222/Realistic_Vision_V1.4", revision="fp16", torch_dtype=torch.float16, scheduler=EulerAncestralDiscreteScheduler)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/txt2img", methods=["POST"])
def txt2img(prompt : str , path : str, steps : int, cfg : float, num_images : int):
    imagesstr = []
    images = txt2img_model(prompt, num_inference_steps=steps, guidance_scale=cfg, num_images_per_prompt=num_images).images
    counter = 0
    for image in images:
        counter+=1
        image.save(f"{path}/Test Image {counter}.png")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        imgstr = base64.b64encode(buffer.getvalue())
        imagesstr.append(imgstr)
    grid = image_grid(images)
    grid.save(f"{path}/Image Grid.png")
    return Response(content=imagesstr, media_type="image/png")