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

models = DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline

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

def load_pipeline(model_id : str, revision : str, torch_dtype : torch, scheduler) -> models:
    distributed_state = PartialState()
    txt2img = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=True)
    choose_scheduler(scheduler, txt2img)
    components = txt2img.components
    img2img = StableDiffusionImg2ImgPipeline(**components)
    inpaint = StableDiffusionInpaintPipeline(**components)
    return txt2img, img2img, inpaint