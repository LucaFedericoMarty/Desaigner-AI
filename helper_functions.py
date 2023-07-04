from diffusers import ( DiffusionPipeline,
StableDiffusionImg2ImgPipeline,
StableDiffusionInpaintPipeline,
StableDiffusionImageVariationPipeline,
EulerAncestralDiscreteScheduler)
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import zipfile
from accelerate import PartialState
import base64
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from fastapi import Response
import json

models = DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionImageVariationPipeline

def weight_keyword(keyword : str, weight : float) -> dict:
    weighted_keyword = {keyword : weight}
    for key, value in weighted_keyword.items():
        string_weighted_keyword = (f'{key} : {value}')
    return string_weighted_keyword

def create_prompt(budget : str, style : str , environment : str, region_weather : str) -> str:
  budget += " budget"
  budget_w = weight_keyword(budget, 0.5)
  environment_w = weight_keyword(environment, 1)
  style_w = weight_keyword(style, 0.6)
  region_weather += " weather"
  region_weather_w = weight_keyword(region_weather, 0.2)
  resolution = "8k"
  picture_style = "hyperrealistic"
  prompt = f"Interior design of a {environment_w}, {style_w}, for a {region_weather_w}, {picture_style}, {budget_w}, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
  return prompt

def image_grid(imgs, rows=2, cols=2):
  # * Get the width and height of the first image
  width, height = imgs[0].size
  # * Create the image grid, with the size of the images
    # * Example: 4 images of 512x512
      # * Height: 512 X 2 = 1024
      # * Width: 512 X 2 = 1024
  grid = Image.new("RGB", size=(cols * width, rows * height))

  # * Paste images into the grid
  for i, img in enumerate(imgs):
      grid.paste(img, box=(i % cols * width, i // cols * height))
  return grid

def choose_scheduler(scheduler_name, model_pipeline):
    if scheduler_name in model_pipeline.scheduler.compatibles:
        model_pipeline.scheduler = scheduler_name.from_config(model_pipeline.scheduler.config)
    else:
      f"The scheduler {scheduler_name} is not compatible with {model_pipeline}"

def load_pipelines(model_id : str, scheduler, **config) -> models:
    txt2img = DiffusionPipeline.from_pretrained(model_id, revision=config.get('revision'), torch_dtype=config.get('torch_dtype'), use_safetensors=True)
    choose_scheduler(scheduler, txt2img)
    #txt2img.enable_xformers_memory_efficient_attention()
    # Workaround for not accepting attention shape using VAE for Flash Attention
    #txt2img.vae.enable_xformers_memory_efficient_attention()
    txt2img.enable_vae_slicing()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt2img.to(device)
    components = txt2img.components
    img2img = StableDiffusionImg2ImgPipeline(**components)
    inpaint = StableDiffusionInpaintPipeline(**components)
    imgvariation = StableDiffusionImageVariationPipeline.from_pretrained('lambdalabs/sd-image-variations-diffusers', revision="v2.0")
    return txt2img, img2img, inpaint, imgvariation

def zip_files(file_objects):
  zip_filename = 'images.zip'
  with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for i, file_obj in enumerate(file_objects):
      zipf.writestr(f'image {i + 1}.png', file_obj.getvalue())
  return zip_filename

def save_images(images : list[Image.Image]):
  file_objects= []
  for num_image in range(len(images)):
    image = images[num_image]
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    file_objects.append(buffer)
  return file_objects

def encode_save_images(images : list[Image.Image]):
  # * Initialize an empty dictionary to store the encoded images:
    # * Example: {Image 1 : 0xbcmnshalla}
  encoded_images_dict = {}
  # * Index each image with respective index 
  for num_image in range(len(images)):
    # * Access an image with its index
    image = images[num_image]
    # * Create a buffer (Space of memory in RAM)
    buffer = BytesIO()
    # * Save each image in memory
    image.save(buffer, format="PNG")
    # * Encode each buffer's value (Image) in base64 to convert it in binary
    binary_image = base64.b64encode(buffer.getvalue())
    # * Open the binary image and read it to convert it to a string
    with open(binary_image, "rb") as img_file:
      binary_string_image = base64.b64encode(img_file.read())
    # * Create each instance of the image with its name as its key and its binary value encoded in base64 in string
    encoded_images_dict[f"Image {num_image + 1}"] = binary_string_image
  # * Convert the image dictionary to a json and return it
  jsonImages = json.dumps(encoded_images_dict)
  return jsonImages