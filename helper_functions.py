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
import json

# * Alias for models

models = DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionImageVariationPipeline

def weight_keyword(keyword : str, weight : float) -> str:
    """Weight each keyword by the given weight"""

    # * Weight the keyword by the given weight in a string format
    return (f'{keyword} : {weight}')

def create_prompt(budget : str, style : str , environment : str, region_weather : str) -> str:
  """Creat an adequate prompt with each keyword weighted"""

  # * Create all the keywords or key phrases to weight
  budget += " budget"
  budget_w = weight_keyword(budget, 0.5)
  environment_w = weight_keyword(environment, 1)
  style_w = weight_keyword(style, 0.6)
  region_weather += " weather"
  region_weather_w = weight_keyword(region_weather, 0.2)

  # * Create the prompt with additional details to improve its performance
  prompt = f"Interior design of a {environment_w}, {style_w}, for a {region_weather_w}, {budget_w}, hyperrealistic, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
  return prompt

def image_grid(imgs, rows=2, cols=2):
  """Generate an image grid given a number of rows and columns"""

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
    """Choose the scheduler given a name and the pipeline desired to change if compatible"""

    # * If the scheduler is compatible with the given pipeline, change the scheduler pipeline
    if scheduler_name in model_pipeline.scheduler.compatibles:
        model_pipeline.scheduler = scheduler_name.from_config(model_pipeline.scheduler.config)
    # * Else, return a message indicating that the scheduler is not compatible with the given pipeline
    else:
      f"The scheduler {scheduler_name} is not compatible with {model_pipeline}"

def load_pipelines(model_id : str, scheduler, **config) -> models:
    """Load the model pipeline and configure it"""

    # * Load the model pipeline txt2img model
    txt2img = DiffusionPipeline.from_pretrained(model_id, revision=config.get('revision'), torch_dtype=config.get('torch_dtype'), use_safetensors=True)
    # * Change the model scheduler
    choose_scheduler(scheduler, txt2img)
    #txt2img.enable_xformers_memory_efficient_attention()
    # Workaround for not accepting attention shape using VAE for Flash Attention
    #txt2img.vae.enable_xformers_memory_efficient_attention()
    # * Configuration for optimization
    txt2img.enable_vae_slicing()
    # * If cuda GPU available, set the device to the GPU. Else, set CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # * Move to the given device
    txt2img.to(device)
    # * Grab the model pipeline components
    components = txt2img.components
    # * Load the img2img, inpaint and image variation model
    img2img = StableDiffusionImg2ImgPipeline(**components)
    inpaint = StableDiffusionInpaintPipeline(**components)
    imgvariation = StableDiffusionImageVariationPipeline.from_pretrained('lambdalabs/sd-image-variations-diffusers', revision="v2.0")
    return txt2img, img2img, inpaint, imgvariation

def zip_images(file_objects : BytesIO):
  """Zip images given their file objects. It returns a zip folder with the corresponding image file name and its value"""

  # * Define the name of the zip file
  zip_filename = 'images.zip'
  # * Open the zip file to write information in it
  with zipfile.ZipFile(zip_filename, 'w') as zipf:
    # * Write the information of the buffer in the zip file  
    for i, file_obj in enumerate(file_objects):
      # * Each file of the zip file will have "Image 'num_image'" as its filename and will write the image information in it 
      zipf.writestr(f'image {i + 1}.png', file_obj.getvalue())
  return zip_filename

def save_images(images : list[Image.Image]) -> BytesIO:
  """Save the images file objects in a list"""

  # * Initialize a list to store the image file objects
  file_objects= []
  # * Index the images
  for image in images:
    # * Create the buffer for storing each image
    buffer = BytesIO()
    # * Save each image in the buffer
    image.save(buffer, format="PNG")
    # * Store the image file objects in the list previously mentioned
    buffer.seek(0)
    file_objects.append(buffer)
  return file_objects

def images_to_b64(images : list[Image.Image]) -> str:
  """Convert a list of images into a JSON string with each image encoded in base64 format"""

  # * Initialize an empty dictionary to store the encoded images:
    # * Example: {Image 1 : 0xbcmnshalla}
  encoded_images_dict = {}
  # * Index each image with its respective index 
  for num_image, image in enumerate(images):
    # * Create a memory buffer (Space of memory in RAM) to store the image data in bytes
    buffer = BytesIO()
    # * Save each image in memory
    image.save(buffer, format="PNG")
    # * Get the image type
    image_type = image.format
    # * Encode each buffer's value (Image in binary) in base64
      # * Buffer.getvalue() gets the image binary data in hex format --> xd9\x80\xb4\xa1\x14I^ \xd3\x94\xd8$\x11\xac\x14\x82\xc2\xe3\x9a\xc1\x80\xe8\x01d\x920\'\x16\x13\x91\x1bJ}\xebI\xa5\xc0\xe3|r\xc3:Y
      # * Then encode this binary data in base64:
        # * Example: "Python"
        # * - Take the ASCII value of each character in the string --> P, y, t, h, o, n are 15, 50, 45, 33, 40, 39
        # * - Calculate the 8-bit binary equivalent of the ASCII values --> 15, 50, 45, 33, 40, 39 are 01010000 01111001 01110100 01101000 01101111 01101110
        # * - Convert the 8-bit chunks into chunks of 6 bits by simply re-grouping the digits --> 01010000 01111001 01110100 01101000 01101111 01101110 are 010100 000111 100101 110100 011010 000110 111101 101110
        # * - Convert the 6-bit binary groups to their respective decimal values. --> 010100 000111 100101 110100 011010 000110 111101 101110 are 20, 7, 37, 52, 26, 6, 61, 46
        # * - Using a base64 encoding table, assign the respective base64 character for each decimal value --> 20, 7, 37, 52, 26, 6, 61, 46 are UHl0aG9u
    encodedB64_image = base64.urlsafe_b64encode(buffer.getvalue())
    # * Decode it in latin1 in order to be a string compatible with JSON
    encodedB64_image_string = encodedB64_image.decode('latin1')
    # * Create each instance of the dictionary with the following values:
      # * Image number
      # * Encoded image string
      # * Image type
    encoded_images_dict["Image number"] = num_image + 1
    encoded_images_dict["Encoded image"] = encodedB64_image_string
    encoded_images_dict["Image type"] = image_type
  # * Convert the image dictionary to a JSON and return it
  jsonImages = json.dumps(encoded_images_dict)
  return jsonImages