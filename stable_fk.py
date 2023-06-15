import os
import requests
import json
import base64

from diffusers import DiffusionPipeline
import torch

def create_prompt(budget, style, enviroment, region_weather):
  budget += " budget"
  budget_w = weight_keyword(budget, 0.6)
  enviroment_w = weight_keyword(enviroment, 0.9)
  style_w = weight_keyword(style, 0.5)
  region_weather += " weather"
  region_weather_w = weight_keyword(region_weather, 0.3)
  resolution = "8k"
  picture_style = "hyperrealistic"
  prompt = f"Interior design of a {enviroment_w}, {style_w}, for a {region_weather_w}, {picture_style}, {budget_w}, in {resolution}"
  return prompt

def weight_keyword(keyword, weight):
    weighted_keyword = {keyword : weight}
    for key, value in weighted_keyword.items():
        string_weighted_keyword = (f'{key} : {value}')
    return string_weighted_keyword

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")

pipeline.to("cuda")

prompt = "Interior design of a living room, art deco style"

image = pipeline(prompt).images[0]

image.save(f"Image 2.png")