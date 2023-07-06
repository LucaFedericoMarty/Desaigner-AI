from dataclasses import Field
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

from fastapi import FastAPI, Response, Request, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Required
from typing import Optional, Annotated

from helper_functions import images_to_b64, weight_keyword, create_prompt, image_grid, choose_scheduler, load_pipelines, zip_images , save_images, models  

# http://127.0.0.1:8000

app = FastAPI(
    title="DesAIgner's Stable Diffusion API",
    description="This API provides the service of creating **images** via **Stable Diffusion pre-trained models**",
    version="0.1.0")

class design(BaseModel):
    #model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: str | None = None # * The | None = None makes the attribute optional 
    steps: int
    guidance_scale : float
    num_images: int
    #input_image: Optional[Image.Image]

txt2img_model, img2img_model, inpaint_model, image_variation_model = load_pipelines(model_id = "SG161222/Realistic_Vision_V1.4", scheduler=EulerAncestralDiscreteScheduler) #revision="fp16", #torch_dtype=torch.float16)

@app.get("/")
def test_api():
    """Root route request to test API operation"""
    return {"welcome_message" : "Welcome to my REST API"}

@app.post("/txt2img")
def txt2img(prompt : Annotated[str , Query(title="Prompt for creating images", description="Text to Image Diffusers usually benefit from a more descriptive prompt, try writing detailed things")], 
            steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
            guidance_scale : Annotated[float, Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
            num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2):
            
    """Text-to-image route request that performs a text-to-image process using a pre-trained Stable Diffusion Model"""

    # * Intiliaze the list of file objects --> Direction in memory of files
    file_objects = []
    # * Create the images using the given prompt and some other parameters
    images = txt2img_model(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images).images
    # * Save each image into a space in memory
    file_objects = save_images(images)
    # * Create an image grid
    grid = image_grid(images)
    # * Zip each image file into a zip folder
    zip_filename = zip_images(file_objects)
    # * Return the zip file with the name 'images.zip' and specify its media type
    return FileResponse(zip_filename, filename='images.zip', media_type='application/zip')

@app.post("/txt2img2")
def txt2imgjson(prompt : Annotated[str , Query(title="Prompt for creating images", description="Text to Image Diffusers usually benefit from a more descriptive prompt, try writing detailed things")], 
            steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
            guidance_scale : Annotated[float, Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
            num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2):
    
    """Text-to-image route request that performs a text-to-image process using a pre-trained Stable Diffusion Model"""

    # * Create the images using the given prompt and some other parameters
    images = txt2img_model(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images).images
    # * Encode the images in base64 and save them to a JSON file
    jsonImages = images_to_b64(images)
    # * Make compatible the JSON with the response
    jsonCompatibleImages = jsonable_encoder(jsonImages)
    # * Create an image grid
    grid = image_grid(images)

    # * Set personalized JSON filename and create the headers for the JSON file
    filename = "base64images.json"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}

    return JSONResponse(content=jsonCompatibleImages, headers=headers)

@app.post("/inpaint")
def inpaint(prompt : Annotated[str , Query(title="Prompt for creating images", description="Text to Image Diffusers usually benefit from a more descriptive prompt, try writing detailed things")],
            input_image : Annotated[UploadFile , Query(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image")], 
            mask_image : Annotated[UploadFile , Query(title="Image mask of the input image", description="This image should be in black and white, and the white parts should be the parts you want to change and the black parts the ones you want to mantain")], 
            steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
            guidance_scale : Annotated[float , Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
            num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2 
            ):

    """Inpainting route request that performs a text-to-image process in the mask of the image using a pre-trained Stable Diffusion Model"""

    # * Create the images using the given prompt and some other parameters
    images = inpaint_model(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance_scale, image=input_image, mask_image=mask_image, num_images_per_prompt=num_images).images
    # * Encode the images in base64 and save them to a JSON file
    jsonImages = images_to_b64(images)
    # * Make compatible the JSON with the response
    jsonCompatibleImages = jsonable_encoder(jsonImages)
    # * Create an image grid
    grid = image_grid(images)

    # * Set personalized JSON filename and create the headers for the JSON file
    filename = "base64images.json"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}

    return JSONResponse(content=jsonCompatibleImages, headers=headers)

@app.post("/image_variation")
def image_variation(prompt : Annotated[str , Query(title="Prompt for creating images", description="Text to Image Diffusers usually benefit from a more descriptive prompt, try writing detailed things")],
            input_image : Annotated[UploadFile , Query(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image")],  
            steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
            guidance_scale : Annotated[float , Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
            num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2 
            ):
    
    """Image variation route request that performs a CLIP process in the input image, that outputs a description of an image that afterwards is used as the prompt to create similar images"""

    # * Create the images using the given prompt and some other parameters
    images = image_variation_model(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance_scale, image=input_image, num_images_per_prompt=num_images).images
    # * Encode the images in base64 and save them to a JSON file
    jsonImages = images_to_b64(images)
    # * Make compatible the JSON with the response
    jsonCompatibleImages = jsonable_encoder(jsonImages)
    # * Create an image grid
    grid = image_grid(images)

    # * Set personalized JSON filename and create the headers for the JSON file
    filename = "base64images.json"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}

    return JSONResponse(content=jsonCompatibleImages, headers=headers)