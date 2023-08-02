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

from fastapi import FastAPI, Response, Request, HTTPException, UploadFile, Query, Body, File, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import Optional, Annotated

from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

from starlette_validation_uploadfile import ValidateUploadFileMiddleware

from functions.helper_functions import images_to_b64, weight_keyword, create_prompt, image_grid, choose_scheduler, load_all_pipelines, load_mlsd_detector ,zip_images , images_to_bytes, images_to_mime, images_to_mime2, models  

# http://127.0.0.1:8000

app = FastAPI(
    title="DesAIgner's Stable Diffusion API",
    description="This API provides the service of creating **images** via **Stable Diffusion pre-trained models**",
    version="0.0.1")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# * Class for counting the process time of the request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = time.strftime("%M:%S", time.gmtime(process_time))
    return response
    
# * Origings for CORS requests    
    
origins = ["http://localhost:8000", "http://localhost:3000"]

# * Adding CORS Class to the Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"], 
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
class design(BaseModel):
    #model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: str | None = None # * The | None = None makes the attribute optional 
    steps: int
    guidance_scale : float
    num_images: int
    #input_image: Optional[Image.Image]

txt2img_model, img2img_model = load_all_pipelines(model_id = "SG161222/Realistic_Vision_V5.0_noVAE", inpaint_model_id = "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/blob/main/Realistic_Vision_V5.1_fp16-no-ema-inpainting.safetensors")
mlsd_detector =  load_mlsd_detector(model_id='lllyasviel/ControlNet')#revision="fp16", #torch_dtype=torch.float16)

@app.get("/")
def test_api():
    """Root route request to test API operation"""
    return {"welcome_message" : "Welcome to my REST API"}

@app.post("/txt2img")
def txt2img(budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
            style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")], 
            environment : Annotated[str , Query(title="Enviroment of the re-design", description="The enviorment you are looking to re-design")],
            region_weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")],
            steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
            guidance_scale : Annotated[float, Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 7, 
            num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2):
            
    """Text-to-image route request that performs a text-to-image process using a pre-trained Stable Diffusion Model"""

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, region_weather=region_weather)
    # * Create the negative prompt for avoiding certain concepts in the photo
    negative_prompt = ("blurry : 1.3, abstract : 1.4, cartoon : 1.4, animated : 1.5, unrealistic : 1.6, watermark : 1.2, signature, two faces, black man, duplicate, copy, multi, two, disfigured, kitsch, ugly, oversaturated, contrast, grain, low resolution, deformed, blurred, bad anatomy, disfigured, badly drawn face, mutation, mutated, extra limb, ugly, bad holding object, badly drawn arms, missing limb, blurred, floating limbs, detached limbs, deformed arms, blurred, out of focus, long neck, long body, ugly, disgusting, badly drawn, childish, disfigured,old ugly, tile, badly drawn arms, badly drawn legs, badly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurred, bad anatomy, blurred, watermark, grainy, signature, clipped, draftbird view, bad proportion, hero, cropped image, overexposed, underexposed, image on TV : 1.5, TV turned on : 1.4")
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

@app.post("/txt2img2")
def txt2imgjson(budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
                style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")], 
                environment : Annotated[str , Query(title="Enviroment of the re-design", description="The enviorment you are looking to re-design")],
                region_weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")], 
                steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
                guidance_scale : Annotated[float, Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
                num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=1, le=6)] = 2):
    
    """Text-to-image route request that performs a text-to-image process using a pre-trained Stable Diffusion Model"""

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, region_weather=region_weather)
    # * Create the negative prompt for avoiding certain concepts in the photo
    negative_prompt = ("blurry, abstract, cartoon, animated, unrealistic, watermark, signature, two faces, black man, duplicate, copy, multi, two, disfigured, kitsch, ugly, oversaturated, contrast, grain, low resolution, deformed, blurred, bad anatomy, disfigured, badly drawn face, mutation, mutated, extra limb, ugly, bad holding object, badly drawn arms, missing limb, blurred, floating limbs, detached limbs, deformed arms, blurred, out of focus, long neck, long body, ugly, disgusting, badly drawn, childish, disfigured,old ugly, tile, badly drawn arms, badly drawn legs, badly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurred, bad anatomy, blurred, watermark, grainy, signature, clipped, draftbird view, bad proportion, hero, cropped image, overexposed, underexposed")
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

@app.post("/txt2img3")
def txt2imgBytes(budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
                style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")], 
                environment : Annotated[str , Query(title="Enviroment of the re-design", description="The enviorment you are looking to re-design")],
                region_weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")], 
                steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
                guidance_scale : Annotated[float, Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
                num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2):
    
    """Text-to-image route request that performs a text-to-image process using a pre-trained Stable Diffusion Model"""

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, region_weather=region_weather)
    # * Create the negative prompt for avoiding certain concepts in the photo
    negative_prompt = ("blurry, abstract, cartoon, animated, unrealistic, watermark, signature, two faces, black man, duplicate, copy, multi, two, disfigured, kitsch, ugly, oversaturated, contrast, grain, low resolution, deformed, blurred, bad anatomy, disfigured, badly drawn face, mutation, mutated, extra limb, ugly, bad holding object, badly drawn arms, missing limb, blurred, floating limbs, detached limbs, deformed arms, blurred, out of focus, long neck, long body, ugly, disgusting, badly drawn, childish, disfigured,old ugly, tile, badly drawn arms, badly drawn legs, badly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurred, bad anatomy, blurred, watermark, grainy, signature, clipped, draftbird view, bad proportion, hero, cropped image, overexposed, underexposed")
    # * Create the images using the given prompt and some other parameters
    images = txt2img_model(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images).images
    # * Images to list of bytes
    imagesBytes = images_to_bytes(images)
    # * Encode the images bytes list
    imagesBytesFinal = [bytes_final for bytes_final in imagesBytes] 
    # * Create an image grid
    grid = image_grid(images)
    print(type(imagesBytes))
    #print(type(imagesBytesEncoded))

    return imagesBytesFinal

@app.post("/txt2imgmime")
def txt2img_mime(budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
                style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")], 
                environment : Annotated[str , Query(title="Enviroment of the re-design", description="The enviorment you are looking to re-design")],
                region_weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")], 
                steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
                guidance_scale : Annotated[float, Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
                num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2):
    
    """Text-to-image route request that performs a text-to-image process using a pre-trained Stable Diffusion Model"""

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, region_weather=region_weather)
    # * Create the negative prompt for avoiding certain concepts in the photo
    negative_prompt = ("blurry, abstract, cartoon, animated, unrealistic, watermark, signature, two faces, black man, duplicate, copy, multi, two, disfigured, kitsch, ugly, oversaturated, contrast, grain, low resolution, deformed, blurred, bad anatomy, disfigured, badly drawn face, mutation, mutated, extra limb, ugly, bad holding object, badly drawn arms, missing limb, blurred, floating limbs, detached limbs, deformed arms, blurred, out of focus, long neck, long body, ugly, disgusting, badly drawn, childish, disfigured,old ugly, tile, badly drawn arms, badly drawn legs, badly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurred, bad anatomy, blurred, watermark, grainy, signature, clipped, draftbird view, bad proportion, hero, cropped image, overexposed, underexposed")
    # * Create the images using the given prompt and some other parameters
    images = txt2img_model(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images).images
    # * Images to list of bytes
    multipart_data = images_to_mime(images)
    # * Create an image grid
    grid = image_grid(images)

    return StreamingResponse(iter(multipart_data), media_type="multipart/related")

@app.post("/img2img")
def img2img(budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
            style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")], 
            environment : Annotated[str , Query(title="Enviroment of the re-design", description="The enviorment you are looking to re-design")],
            region_weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")],
            input_image : Annotated[UploadFile, File(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image")], 
            steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
            guidance_scale : Annotated[float, Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
            num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2):
    
    """Image-to-image route request that performs a image-to-image process using a pre-trained Stable Diffusion Model"""

    if input_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail=f"File type of {input_image.content_type} is not valid")

    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, region_weather=region_weather)
    # * Create the negative prompt for avoiding certain concepts in the photo
    negative_prompt = ("blurry, abstract, cartoon, animated, unrealistic, watermark, signature, two faces, black man, duplicate, copy, multi, two, disfigured, kitsch, ugly, oversaturated, contrast, grain, low resolution, deformed, blurred, bad anatomy, disfigured, badly drawn face, mutation, mutated, extra limb, ugly, bad holding object, badly drawn arms, missing limb, blurred, floating limbs, detached limbs, deformed arms, blurred, out of focus, long neck, long body, ugly, disgusting, badly drawn, childish, disfigured,old ugly, tile, badly drawn arms, badly drawn legs, badly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurred, bad anatomy, blurred, watermark, grainy, signature, clipped, draftbird view, bad proportion, hero, cropped image, overexposed, underexposed")
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

@app.post("/inpaint")
def inpaint(budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")],
            style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")], 
            environment : Annotated[str , Query(title="Enviroment of the re-design", description="The enviorment you are looking to re-design")],
            region_weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")],
            input_image : Annotated[UploadFile , File(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image")], 
            mask_image : Annotated[UploadFile , File(title="Image mask of the input image", description="This image should be in black and white, and the white parts should be the parts you want to change and the black parts the ones you want to mantain")], 
            steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20, 
            guidance_scale : Annotated[float , Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5, 
            num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2 
            ):

    """Inpainting route request that performs a text-to-image process in the mask of the image using a pre-trained Stable Diffusion Model"""

    if input_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail=f"File type of {input_image.content_type} is not valid")
    
    if mask_image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail=f"File type of {mask_image.content_type} is not valid")
    
    # * Create the prompt for creating the image
    prompt = create_prompt(budget=budget, style=style, environment=environment, region_weather=region_weather)
    # * Create the negative prompt for avoiding certain concepts in the photo
    negative_prompt = ("blurry, abstract, cartoon, animated, unrealistic, watermark, signature, two faces, black man, duplicate, copy, multi, two, disfigured, kitsch, ugly, oversaturated, contrast, grain, low resolution, deformed, blurred, bad anatomy, disfigured, badly drawn face, mutation, mutated, extra limb, ugly, bad holding object, badly drawn arms, missing limb, blurred, floating limbs, detached limbs, deformed arms, blurred, out of focus, long neck, long body, ugly, disgusting, badly drawn, childish, disfigured,old ugly, tile, badly drawn arms, badly drawn legs, badly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurred, bad anatomy, blurred, watermark, grainy, signature, clipped, draftbird view, bad proportion, hero, cropped image, overexposed, underexposed")
    # * Access the file object of the input image and get its contents
    input_image_file_object_content = input_image.file.read()
    # * Create a BytesIO in-memory buffer of the bytes of the image and use it like a file object in order to create a PIL.Image object
    input_img = Image.open(BytesIO(input_image_file_object_content))
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

@app.post("/image_variation")
def image_variation(prompt : Annotated[str , Query(title="Prompt for creating images", description="Text to Image Diffusers usually benefit from a more descriptive prompt, try writing detailed things")],
            input_image : Annotated[UploadFile , Body(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image")],  
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