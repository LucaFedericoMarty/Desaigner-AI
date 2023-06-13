import os
import requests
import json
import base64

from diffusers import DiffusionPipeline
import torch

def create_prompt(price, style, enviroment):
    enviroment_w = weight_keyword(enviroment, 0.9)
    style_w = weight_keyword(style, 0.5)
    resolution = "8k"
    picture_style = "hyperrealistic"
    prompt = f"Interior design of a {enviroment_w}, {style}"
    print(prompt)
    return []

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