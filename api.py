import os
from urllib import response
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

from fastapi import FastAPI, Response, Request, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional

from helper_functions import weight_keyword, create_prompt, image_grid, choose_scheduler, load_pipelines, zipfiles, zip_files , save_images, models  

# http://127.0.0.1:8000

app = FastAPI(
    title="DesAIgner's Stable Diffusion API",
    description="This API provides the service of creating **images** via **Stable Diffusion pre-trained models**",
    version="0.1.0")

class design(BaseModel):
    #model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: Optional[str]
    steps: int
    guidance_scale : float
    num_images: int
    #input_image: Optional[Image.Image]

txt2img_model, img2img_model, inpaint_model, image_variation_model = load_pipelines(model_id = "SG161222/Realistic_Vision_V1.4", scheduler=EulerAncestralDiscreteScheduler) #revision="fp16", #torch_dtype=torch.float16)

@app.get("/")
def test_api():
    """Root route query to test API operation"""
    return {"welcome_message" : "Welcome to my REST API"}

@app.post("/txt2img")
def txt2img(prompt : str, steps : int, guidance_scale : float, num_images : int):
    file_objects = []
    images = txt2img_model(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images).images
    file_objects = save_images(images)
    grid = image_grid(images)
    zip_filename = zip_files(file_objects)
    return FileResponse(zip_filename, filename='images.zip', media_type='application/zip')

@app.post("/txt2img2")
def txt2img2(prompt : str, steps : int, guidance_scale : float, num_images : int):
    paths = []
    images = txt2img_model(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images).images
    for num_image in range(len(images)):
        image = images[num_image]
        buffer = BytesIO()
        path = f"Image {num_image}.png"
        image.save(path, format="PNG")
        imgstr = base64.b64encode(buffer.getvalue())
        paths.append(path)
    grid = image_grid(images)
    return zipfiles(paths)

@app.post("/inpaint")
def inpaint(prompt : str , steps : int, guidance_scale : float, num_images : int, input_image : UploadFile, mask_image : UploadFile):
    images = inpaint_model(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance_scale, image=input_image, mask_image=mask_image, num_images_per_prompt=num_images).images
    file_objects = save_images(images)
    grid = image_grid(images)
    zip_filename = zip_files(file_objects)
    return FileResponse(zip_filename, filename='images.zip', media_type='application/zip')

@app.post("/image_variation")
def inpaint(prompt : str , steps : int, guidance_scale : float, num_images : int, input_image : UploadFile):
    images = image_variation_model(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance_scale, image=input_image, num_images_per_prompt=num_images).images
    file_objects = save_images(images)
    grid = image_grid(images)
    zip_filename = zip_files(file_objects)
    return FileResponse(zip_filename, filename='images.zip', media_type='application/zip')

