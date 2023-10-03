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
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv, find_dotenv

from fastapi import FastAPI, Response, Request, HTTPException, status, UploadFile, Query, Body, File, Depends, Security, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Annotated

from huggingface_hub import hf_hub_download, snapshot_download

from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

from starlette_validation_uploadfile import ValidateUploadFileMiddleware
from starlette.status import HTTP_413_REQUEST_ENTITY_TOO_LARGE, HTTP_415_UNSUPPORTED_MEDIA_TYPE

from fastapi.security.api_key import APIKey
from api.auth.auth import get_api_key

from functions.helper_functions import images_to_b64, images_to_b64_v2, weight_keyword, create_prompt, image_grid, choose_scheduler, load_all_pipelines, load_mlsd_detector ,zip_images , images_to_bytes, images_to_mime, images_to_mime2, size_upload_files, models  

from api.schemas import Txt2ImgParams, Img2ImgParams, InpaintParams, ImageResponse, ImageV2Response

MB_IN_BYTES = 1048576

# http://127.0.0.1:8000

# * Find the path of the dotenv file
dotenv_path = find_dotenv(filename='.env', raise_error_if_not_found=True)

# * Load the enviromental variables from the dotenv path
load_dotenv(dotenv_path)

# * Load the HF token from the dotenv file
HF_TOKEN = os.getenv(key='HF_TOKEN')

tags_metadata = [
    {
        "name": "test",
        "description": "Operations for testing security and API availability",
    },

    {
        "name": "text2image",
        "description": "Creating a new image from user preferences, without an image",
    },

    {
        "name": "image2image",
        "description": "Creating a similar image from user preferences and user's photo",
    },

    {
        "name": "inpaint",
        "description": "Creating a similar image from user preferences and selected parts of user's photo",
    },

    {
        "name": "old",
        "description": "Old and deprecated endpoints",
    },

    {
        "name": "working",
        "description": "Currently used and active endpoints",
    },
]

app = FastAPI(
    title="DesAIgner's Stable Diffusion API",
    description="This API provides the service of creating **images** via **Stable Diffusion pre-trained models**",
    version="0.0.1",
    openapi_tags=tags_metadata,
    )

# * Class for counting the process time of the request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = time.strftime("%M:%S", time.gmtime(process_time))
    return response
    
# * Origings for CORS requests    
    
origins = ["http://localhost:3000", "http://localhost:3000/", "http://localhost:3000/create"]

# * Adding CORS Class to the Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"], 
    allow_credentials=True)

# TODO: Check if use this middleware instead of manually creating HTTP Exceptions

"""

app.add_middleware(
        ValidateUploadFileMiddleware,
        app_path=[
            "/img2img",
            "/inpaint",
        ],
        max_size=16777216,
        file_type=["image/png", "image/jpeg"]
)

"""

# * Download all the files necessary to excute the API
snapshot_download(repo_id="SG161222/Realistic_Vision_V5.1_noVAE", ignore_patterns=["*.gitattriutes", "*.md", "*.ckpt"], allow_patterns=["*.json", "*.txt",  "scheduler/*", "text_encoder/*", "tokenizer/*", "unet/*", "vae/*"], token=HF_TOKEN)
inpaint_model_path = hf_hub_download(repo_id="SG161222/Realistic_Vision_V5.1_noVAE", filename="Realistic_Vision_V5.1_fp16-no-ema-inpainting.safetensors")
snapshot_download(repo_id="lllyasviel/control_v11p_sd15_mlsd", ignore_patterns=["*.gitattriutes", "*.md", "*.bin", "*.py", "*.png"], allow_patterns=["*.json", "*diffusion_pytorch_model.safetensors"], token=HF_TOKEN)
mlsd_detector_path = hf_hub_download(repo_id="lllyasviel/ControlNet", filename="./annotator/ckpts/mlsd_large_512_fp32.pth")

# * Load the models
txt2img_model, img2img_model, inpaint_model = load_all_pipelines(model_id = "SG161222/Realistic_Vision_V5.1_noVAE", inpaint_model_id = inpaint_model_path, controlnet_model="lllyasviel/control_v11p_sd15_mlsd")
mlsd_detector =  load_mlsd_detector(model_id="lllyasviel/ControlNet")#revision="fp16", #torch_dtype=torch.float16)

# * Create the negative prompt for avoiding certain concepts in the photo
negative_prompt = ("blurry, abstract : 1.3, cartoon : 1.3, animated : 1.5, unrealistic : 1.6, watermark, signature, two faces, duplicate, copy, multi, two, disfigured, kitsch, ugly, oversaturated, contrast, grain, low resolution, deformed, blurred, bad anatomy, disfigured, badly drawn face, mutation, mutated, extra limb, ugly, bad holding object, badly drawn arms, missing limb, blurred, floating limbs, detached limbs, deformed arms, blurred, out of focus, long neck, long body, ugly, disgusting, badly drawn, childish, disfigured,old ugly, tile, badly drawn arms, badly drawn legs, badly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurred, bad anatomy, blurred, watermark, grainy, signature, clipped, draftbird view, bad proportion, cropped image, overexposed, underexposed, (image on TV : 1.5), (TV turned on : 1.4)")

@app.get("/", tags=["test"])
def test_api():
    """Root route request to test API operation"""
    return {"welcome_message" : "Welcome to my REST API"}


# * Lockedown Route
@app.get("/secure", tags=["test"])
async def info(api_key: APIKey = Security(get_api_key)):
    """A private endpoint that requires a valid API key to be provided."""
    return {
        "validated api key": api_key
    }

# * Open Route
@app.get("/open", tags=["test"])
async def info():
    """An open endpoint that does not require authentication"""
    return {
        "default variable": "Open Route"
    }

@app.post("/upload-image", tags=["test"])
async def upload(image_file: UploadFile = File(title="Image file to test uploading"),api_key: APIKey = Security(get_api_key)):
    """A private endpoint to test uploading images."""
    return {
        "Image file": image_file
    }


@app.post("/txt2img/v1/v1", tags=["text2image", "old"])
def txt2img(budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
            style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")],
            environment : Annotated[str , Query(title="Environment of the re-design", description="The environment you are looking to re-design")],
            weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")],
            disability : Annotated[str , Query(title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty")],
            steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20,
            guidance_scale : Annotated[float, Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 7,
            num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2,
            api_key: APIKey = Security(get_api_key),):
            
    """Text-to-image route request that performs a text-to-image process using a pre-trained Stable Diffusion Model"""

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, weather=weather, disability=disability)
    # * Intiliaze the list of file objects --> Direction in memory of files
    file_objects = []
    # * Create the images using the given prompt and some other parameters
    images = txt2img_model(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images, generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")).images
    # * Save each image into a space in memory
    file_objects = images_to_bytes(images)
    # * Create an image grid
    grid = image_grid(images)
    # * Zip each image file into a zip folder
    zip_filename = zip_images(file_objects)
    # * Return the zip file with the name 'images.zip' and specify its media type
    return FileResponse(zip_filename, filename='images.zip', media_type='application/zip')

@app.post("/txt2img/v1/v2", tags=["text2image", "old"])
def txt2img_json(budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
                style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")],
                environment : Annotated[str , Query(title="Environment of the re-design", description="The environment you are looking to re-design")],
                weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")],
                disability : Annotated[str , Query(title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty")],
                steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20,
                guidance_scale : Annotated[float, Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 7,
                num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2,
                api_key: APIKey = Security(get_api_key),):
    
    """Text-to-image route request that performs a text-to-image process using a pre-trained Stable Diffusion Model"""

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, weather=weather)
    # * Create the images using the given prompt and some other parameters
    images = txt2img_model(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images).images
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

@app.post("/txt2img/v2/v1", response_model=ImageResponse, tags=["text2image", "working"])
def txt2imgclass(params: Txt2ImgParams, api_key: APIKey = Security(get_api_key)):
    
    """Text-to-image route request that performs a text-to-image process using a pre-trained Stable Diffusion Model"""

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=params.budget, style=params.style, environment=params.environment, weather=params.weather, disability=params.disability)
    # * Create the images using the given prompt and some other parameters
    images = txt2img_model(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=params.steps, guidance_scale=params.guidance_scale, num_images_per_prompt=params.num_images).images
    # * Encode the images in base64 and save them to a JSON file
    b64_images = images_to_b64_v2(images)
    # * Create an image grid
    grid = image_grid(images)

    return ImageResponse(images=b64_images)

@app.post("/txt2img/v1/v3", response_model=ImageV2Response, tags=["text2image", "old"])
def txt2img_bytes(params: Txt2ImgParams, api_key: APIKey = Security(get_api_key)):
    
    """Text-to-image route request that performs a text-to-image process using a pre-trained Stable Diffusion Model"""

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=params.budget, style=params.style, environment=params.environment, weather=params.weather, disability=params.disability)
    # * Create the images using the given prompt and some other parameters
    images = txt2img_model(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=params.steps, guidance_scale=params.guidance_scale, num_images_per_prompt=params.num_images).images
    # * Images to list of bytes
    images_bytes = images_to_bytes(images)
    # * Create list of responses of images bytes using list comprehension
    image_responses = [Response(content=image_bytes, media_type="image/jpeg") for image_bytes in images_bytes]
    print(type(image_responses))

    return ImageV2Response(images=image_responses)

@app.post("/txt2img/v1/v4", tags=["text2image", "old"])
def txt2img_mime(budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
                style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")],
                environment : Annotated[str , Query(title="Environment of the re-design", description="The environment you are looking to re-design")],
                weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")],
                disability : Annotated[str , Query(title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty")],
                steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20,
                guidance_scale : Annotated[float, Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 7,
                num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2,
                api_key: APIKey = Security(get_api_key),):
    
    """Text-to-image route request that performs a text-to-image process using a pre-trained Stable Diffusion Model"""

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, weather=weather, disability=disability)
    # * Create the images using the given prompt and some other parameters
    images = txt2img_model(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images).images
    # * Images to list of bytes
    multipart_data = images_to_mime(images)
    # * Create an image grid
    grid = image_grid(images)

    return StreamingResponse(iter(multipart_data), media_type="multipart/related")

@app.post("/img2img/v1", tags=["image2image", "old"])
def img2img(budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
            style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")],
            environment : Annotated[str , Query(title="Environment of the re-design", description="The environment you are looking to re-design")],
            weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")],
            disability : Annotated[str , Query(title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty")],
            input_image : Annotated[UploadFile, File(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image")],
            steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
            guidance_scale : Annotated[float, Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
            num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2,
            api_key: str = Security(get_api_key),):
    
    """Image-to-image route request that performs a image-to-image process using a pre-trained Stable Diffusion Model"""

    if input_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"File type of {input_image.content_type} is not valid")

    # * Get the file size
    file_size = size_upload_files(input_image)

    # * Raise HTTP Exception in case the file is too large
    if file_size > 5 * MB_IN_BYTES:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, weather=weather, disability=disability)
    # * Move the cursor to the beginning of the file
    input_image.file.seek(0)
    # * Access the file object and get its contents
    input_image_file_object_content = input_image.file.read()
    # * Create a BytesIO in-memory buffer of the bytes of the image and use it like a file object in order to create a PIL.Image object
    input_img = Image.open(BytesIO(input_image_file_object_content))
    # * Convert the image to mlsd line detector format
    input_image_final = mlsd_detector(input_img)
    # * Create the images using the given prompt and some other parameters
    images = img2img_model(prompt=prompt, negative_prompt=negative_prompt, image=input_image_final, controlnet_conditioning_scale = 1.0, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images).images
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

@app.post("/img2img/v2", response_model= ImageResponse, tags=["image2image", "old"])
def img2img(params: Img2ImgParams = Depends(),
            input_image : UploadFile = File(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image"),
            api_key: APIKey = Security(get_api_key),
            ):
    
    """Image-to-image route request that performs a image-to-image process using a pre-trained Stable Diffusion Model"""

    if input_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"File type of {input_image.content_type} is not valid")

    # * Get the file size
    file_size = size_upload_files(input_image)

    # * Raise HTTP Exception in case the file is too large
    if file_size > 5 * MB_IN_BYTES:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=params.budget, style=params.style, environment=params.environment, weather=params.weather, disability=params.disability)
    # * Move the cursor to the beginning of the file
    input_image.file.seek(0)
    # * Access the file object and get its contents
    input_image_file_object_content = input_image.file.read()
    # * Create a BytesIO in-memory buffer of the bytes of the image and use it like a file object in order to create a PIL.Image object
    input_img = Image.open(BytesIO(input_image_file_object_content))
    # * Convert the image to mlsd line detector format
    input_image_final = mlsd_detector(input_img)
    # * Create the images using the given prompt and some other parameters
    images = img2img_model(prompt=prompt, negative_prompt=negative_prompt, image=input_image_final, controlnet_conditioning_scale = params.controlnet_conditioning_scale, num_inference_steps=params.steps, guidance_scale=params.guidance_scale, num_images_per_prompt=params.num_images).images
    # * Encode the images in base64 and save them to a JSON file
    b64Images = images_to_b64_v2(images)
    # * Create an image grid
    grid = image_grid(images)

    return ImageResponse(images=b64Images)

@app.post("/img2img/v3", tags=["image2image", "working"])
def img2img_form(budget : Annotated[str , Form(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
            style : Annotated[str , Form(title="Style of the re-design", description="Choose any interior design style that best suits your desires")],
            environment : Annotated[str , Form(title="Environment of the re-design", description="The environment you are looking to re-design")],
            weather : Annotated[str , Form(title="Weather of the region", description="The typical weather you of the region you are living")],
            disability : Annotated[str , Form(title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty")],
            input_image : Annotated[UploadFile, File(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image")],
            steps : Annotated[int , Form(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=1, le=50)] = 20, 
            guidance_scale : Annotated[float, Form(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 7, 
            num_images : Annotated[int , Form(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=1, le=4)] = 1,
            api_key: str = Security(get_api_key),):
    
    """Image-to-image route request that performs a image-to-image process using a pre-trained Stable Diffusion Model"""

    if input_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"File type of {input_image.content_type} is not valid")

    # * Get the file size
    file_size = size_upload_files(input_image)

    # * Raise HTTP Exception in case the file is too large
    if file_size > 5 * MB_IN_BYTES:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, weather=weather, disability=disability)
    # * Move the cursor to the beginning of the file
    input_image.file.seek(0)
    # * Access the file object and get its contents
    input_image_file_object_content = input_image.file.read()
    # * Create a BytesIO in-memory buffer of the bytes of the image and use it like a file object in order to create a PIL.Image object
    input_img = Image.open(BytesIO(input_image_file_object_content))
    # * Convert the image to mlsd line detector format
    input_image_final = mlsd_detector(input_img)
    # * Create the images using the given prompt and some other parameters
    images = img2img_model(prompt=prompt, negative_prompt=negative_prompt, image=input_image_final, controlnet_conditioning_scale = 1.0, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images).images
    # * Encode the images in base64 and save them to a JSON file
    b64_images = images_to_b64_v2(images)
    # * Create an image grid
    grid = image_grid(images)

    return ImageResponse(images=b64_images)

@app.post("/inpaint/v1", tags=["inpaint", "old"])
def inpaint(budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
            style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")],
            environment : Annotated[str , Query(title="Environment of the re-design", description="The environment you are looking to re-design")],
            weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")],
            disability : Annotated[str , Query(title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty")],
            input_image : Annotated[UploadFile , File(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image")], 
            mask_image : Annotated[UploadFile , File(title="Image mask of the input image", description="This image should be in black and white, and the white parts should be the parts you want to change and the black parts the ones you want to mantain")], 
            steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
            guidance_scale : Annotated[float , Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
            num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2, 
            api_key: APIKey = Security(get_api_key),):

    """Inpainting route request that performs a text-to-image process in the mask of the image using a pre-trained Stable Diffusion Model"""

    if input_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"File type of {input_image.content_type} is not valid")
    
    if mask_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"File type of {mask_image.content_type} is not valid")
    
    # * Get the file size
    file_size = size_upload_files(input_image)

    # * Raise HTTP Exception in case the file is too large
    if file_size > 5 * MB_IN_BYTES:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")
    
    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, weather=weather, disability=disability)
    # * Move the cursor to the beginning of the file
    input_image.file.seek(0)
    # * Access the file object of the input image and get its contents
    input_image_file_object_content = input_image.file.read()
    # * Create a BytesIO in-memory buffer of the bytes of the image and use it like a file object in order to create a PIL.Image object
    input_img = Image.open(BytesIO(input_image_file_object_content))
    # * Move the cursor to the beginning of the file
    mask_image.file.seek(0)
    # * Access the file object of the maske image and get its contents
    mask_image_file_object_content = mask_image.file.read()
    # * Create a BytesIO in-memory buffer of the bytes of the image and use it like a file object in order to create a PIL.Image object
    mask_img = Image.open(BytesIO(mask_image_file_object_content))
    # * Create the images using the given prompt and some other parameters
    images = inpaint_model(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=guidance_scale, image=input_img, mask_image=mask_img, num_images_per_prompt=num_images).images
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

@app.post("/inpaint/v2", response_model=ImageResponse ,tags=["inpaint", "old"])
def inpaint(params : InpaintParams = Depends(),
            input_image : UploadFile = File(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image"),
            mask_image : UploadFile = File(title="Image mask of the input image", description="This image should be in black and white, and the white parts should be the parts you want to change and the black parts the ones you want to maintain"),
            api_key: APIKey = Security(get_api_key),):

    """Inpainting route request that performs a text-to-image process in the mask of the image using a pre-trained Stable Diffusion Model"""

    if input_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"File type of {input_image.content_type} is not valid")
    
    if mask_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"File type of {mask_image.content_type} is not valid")
    
    # * Get the file size
    file_size = size_upload_files(input_image)

    # * Raise HTTP Exception in case the file is too large
    if file_size > 5 * MB_IN_BYTES:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")
    
    # * Create the prompt for creating the image
    prompt = create_prompt(budget=params.budget, style=params.style, environment=params.environment, weather=params.weather, disability=params.disability)
    # * Move the cursor to the beginning of the file
    input_image.file.seek(0)
    # * Access the file object of the input image and get its contents
    input_image_file_object_content = input_image.file.read()
    # * Create a BytesIO in-memory buffer of the bytes of the image and use it like a file object in order to create a PIL.Image object
    input_img = Image.open(BytesIO(input_image_file_object_content))
    # * Move the cursor to the beginning of the file
    mask_image.file.seek(0)
    # * Access the file object of the maske image and get its contents
    mask_image_file_object_content = mask_image.file.read()
    # * Create a BytesIO in-memory buffer of the bytes of the image and use it like a file object in order to create a PIL.Image object
    mask_img = Image.open(BytesIO(mask_image_file_object_content))
    # * Create the images using the given prompt and some other parameters
    images = inpaint_model(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=params.steps, guidance_scale=params.guidance_scale, image=input_img, mask_image=mask_img, num_images_per_prompt=params.num_images).images
    # * Encode the images in base64 and save them to a JSON file
    b64Images = images_to_b64_v2(images)
    # * Create an image grid
    grid = image_grid(images)

    return ImageResponse(images=b64Images)

@app.post("/inpaint/v3", tags=["inpaint", "working"])
def inpaint(budget : Annotated[str , Form(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
            style : Annotated[str , Form(title="Style of the re-design", description="Choose any interior design style that best suits your desires")],
            environment : Annotated[str , Form(title="Environment of the re-design", description="The environment you are looking to re-design")],
            weather : Annotated[str , Form(title="Weather of the region", description="The typical weather you of the region you are living")],
            disability : Annotated[str , Form(title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty")],
            input_image : Annotated[UploadFile , File(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image")], 
            mask_image : Annotated[UploadFile , File(title="Image mask of the input image", description="This image should be in black and white, and the white parts should be the parts you want to change and the black parts the ones you want to mantain")], 
            steps : Annotated[int , Form(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
            guidance_scale : Annotated[float , Form(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
            num_images : Annotated[int , Form(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2, 
            api_key: APIKey = Security(get_api_key),):

    """Inpainting route request that performs a text-to-image process in the mask of the image using a pre-trained Stable Diffusion Model"""

    if input_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"File type of {input_image.content_type} is not valid")
    
    if mask_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"File type of {mask_image.content_type} is not valid")
    
    # * Get the file size
    file_size = size_upload_files(input_image)

    # * Raise HTTP Exception in case the file is too large
    if file_size > 5 * MB_IN_BYTES:
        raise HTTPException(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")
    
    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, weather=weather, disability=disability)
    # * Move the cursor to the beginning of the file
    input_image.file.seek(0)
    # * Access the file object of the input image and get its contents
    input_image_file_object_content = input_image.file.read()
    # * Create a BytesIO in-memory buffer of the bytes of the image and use it like a file object in order to create a PIL.Image object
    input_img = Image.open(BytesIO(input_image_file_object_content))
    # * Move the cursor to the beginning of the file
    mask_image.file.seek(0)
    # * Access the file object of the maske image and get its contents
    mask_image_file_object_content = mask_image.file.read()
    # * Create a BytesIO in-memory buffer of the bytes of the image and use it like a file object in order to create a PIL.Image object
    mask_img = Image.open(BytesIO(mask_image_file_object_content))
    # * Create the images using the given prompt and some other parameters
    images = inpaint_model(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=guidance_scale, image=input_img, mask_image=mask_img, num_images_per_prompt=num_images).images
    # * Encode the images in base64 and save them to a JSON file
    b64_images = images_to_b64_v2(images)
    # * Create an image grid
    grid = image_grid(images)

    return ImageResponse(images=b64_images)

@app.post("/image_variation")
def image_variation(budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
                    style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")],
                    environment : Annotated[str , Query(title="Environment of the re-design", description="The environment you are looking to re-design")],
                    weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")],
                    disability : Annotated[str , Query(title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty")],
                    input_image : Annotated[UploadFile , Body(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image")],  
                    steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
                    guidance_scale : Annotated[float , Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
                    num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2, 
                    api_key: str = Security(get_api_key),):
    
    """Image variation route request that performs a CLIP process in the input image, that outputs a description of an image that afterwards is used as the prompt to create similar images"""

    # * Create the images using the given prompt and some other parameters
    prompt = create_prompt(budget=budget, style=style, environment=environment, weather=weather, disability=disability)
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