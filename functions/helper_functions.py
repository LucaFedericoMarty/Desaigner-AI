from diffusers import (DiffusionPipeline,
StableDiffusionPipeline,
StableDiffusionImg2ImgPipeline,
StableDiffusionInpaintPipeline,
StableDiffusionImageVariationPipeline,
EulerAncestralDiscreteScheduler,
UniPCMultistepScheduler,
StableDiffusionControlNetPipeline,
ControlNetModel,
DPMSolverMultistepScheduler)
from controlnet_aux import MLSDdetector
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


from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# * Alias for models

models = DiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionInpaintPipeline

def lowercase_first_letter(word : str) -> str:
  """Lowercase the first letter of a word"""
  return word[0].lower() + word[1:]

def weight_keyword(keyword : str, weight : float) -> str:
    """Weight each keyword by the given weight"""

    # * Weight the keyword by the given weight in a string format
    return (f'({keyword} : {weight})')

def create_prompt(budget : str, style : str , environment : str, weather : str, disability : str = "") -> str:
  """Creat an adequate prompt with each keyword weighted"""

  if len(disability) > 0:
    disability = lowercase_first_letter(disability)

  # * Create all the keywords or key phrases to weight
  budget += " budget"
  budget_w = weight_keyword(budget, 0.7)
  environment_w = weight_keyword(environment, 1.2)
  style += " style"
  style_w = weight_keyword(style, 1.7)
  weather += " weather"
  weather_w = weight_keyword(weather, 0.4)
  disability_new = f'adapted and usable for {disability}'
  disability_w = weight_keyword(disability_new, 0.5)

  # * Create the prompt with additional details to improve its performance
  normal_prompt = f"RAW photo, masterpiece, interior design, {environment_w}, {style_w}, {weather_w}, {budget_w}, ultra realistic render, 3D art, daylight, hyperrealistic, photorealistic, ultradetailed, 8k, soft lighting, high quality, film grain, Fujifilm XT3"
  disability_prompt = f"RAW photo, masterpiece, interior design, {environment_w}, {style_w}, {weather_w}, {budget_w}, {disability_w}, ultra realistic render, 3D art, daylight, hyperrealistic, photorealistic, ultradetailed, 8k, soft lighting, high quality, film grain, Fujifilm XT3"

  if not disability:
    prompt = normal_prompt
  else:
    prompt = disability_prompt

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

def load_all_pipelines(model_id: str, inpaint_model_id : str,  txt2img_scheduler = DPMSolverMultistepScheduler, img2img_scheduler = UniPCMultistepScheduler, controlnet_model = "lllyasviel/control_v11p_sd15_mlsd") -> models:
    """Load the model pipeline and configure it"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # * Set the torch dtype to float16 if GPU are available, else set to float16, since float16 is not compatible with CPUs.
    torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32)
    pipelines = []

    # * Load the MLSD controlnet model pipeline

    controlnet = ControlNetModel.from_pretrained(
    controlnet_model, torch_dtype=torch_dtype, use_safetensos=True)

    # TODO: Load Stable Diffusion pipeline with OpenVINO or ONNX

    # * Load the model pipeline txt2img model
    with torch.no_grad():
      txt2img = StableDiffusionPipeline.from_pretrained(
        model_id,
        custom_pipeline="lpw_stable_diffusion",
        torch_dtype=torch_dtype,
        #revision='fp16',
        use_safetensors=True,
        )
      choose_scheduler(txt2img_scheduler, txt2img)
      txt2img.enable_vae_slicing()
      txt2img.enable_attention_slicing()
      if torch.cuda.is_available():
        txt2img.enable_xformers_memory_efficient_attention()
        txt2img.enable_model_cpu_offload()
      #components = txt2img.components
      pipelines.append(txt2img)

    # * Load the txt2img guided with controlnet model used as an img2img model
    with torch.no_grad():
      img2img = StableDiffusionControlNetPipeline.from_pretrained(
          model_id,
          #custom_pipeline="lpw_stable_diffusion",
          torch_dtype=torch_dtype,
          #revision='fp16',
          use_safetensors=True,
          controlnet=controlnet)
      choose_scheduler(img2img_scheduler, img2img)
      img2img.enable_vae_slicing()
      img2img.enable_attention_slicing()
      if torch.cuda.is_available():
        img2img.enable_xformers_memory_efficient_attention()
        img2img.enable_model_cpu_offload()
      pipelines.append(img2img)

    """
    # * Load the inpaint model
    with torch.no_grad():
      inpaint = StableDiffusionInpaintPipeline.from_single_file(
        inpaint_model_id,
        #custom_pipeline="lpw_stable_diffusion",
        #torch_dtype=torch_dtype,
        )
      choose_scheduler(txt2img_scheduler, inpaint)
      #inpaint.enable_vae_slicing()
      inpaint.enable_attention_slicing()
      if torch.cuda.is_available():
        inpaint.enable_xformers_memory_efficient_attention()
        inpaint.enable_model_cpu_offload()
      pipelines.append(inpaint)

    """

    # * Clear intermediate variables
    del txt2img, img2img

    return tuple(pipelines)

def load_mlsd_detector(model_id : str):
   return MLSDdetector.from_pretrained(model_id)

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

def images_to_bytes(images : list[Image.Image]) -> list[BytesIO]:
  """Convert images to a list of bytes"""

  # * Initialize a list to store the image file objects
  file_objects= []
  # * Index the images
  for image in images:
    # * Create the buffer for storing each image
    buffer = BytesIO()
    # * Save each image in the buffer
    image.save(buffer, format="PNG")
    # * Store the image file objects in the list previously mentioned
    #buffer.seek(0)
    file_objects.append(buffer.getvalue().decode("latin-1"))
  return file_objects

def images_to_mime(images : list[Image.Image]):
  """Convert images to mime objects"""

  # * Create a Multipart message
  multipart_data = MIMEMultipart()
  # * Index the images
  for counter, image in enumerate(images):
    # * Create the buffer for storing each image
    buffer = BytesIO()
    # * Save each image in the buffer
    image.save(buffer, format="PNG")
    # * Store the image file objects in the list previously mentioned
    buffer.seek(0)
    # * Create a MIMEImage object for each image
    image_part = MIMEImage(buffer.read(), _subtype="png")
    # * Set filename
    image_part.add_header("Content-Disposition", f'inline; filename="Image {counter}.png"')
    # * Attach the image to the Multipart message    
    multipart_data.attach(image_part)
  print(multipart_data)
  print(multipart_data.as_bytes())  
  return multipart_data.as_bytes()

def images_to_mime2(images : list[Image.Image]) -> list[BytesIO]:
  """Convert images to mime objects"""

   # Create the multipart content
  content = b''

  # * Index the images
  for image in images:
    # * Create the buffer for storing each image
    buffer = BytesIO()
    # * Save each image in the buffer
    image.save(buffer, format="PNG")
    # * Store the image file objects in the list previously mentioned
    buffer.seek(0)
    # Add the image data to the multipart content
    content += b'--boundary\r\n'
    content += b'Content-Disposition: form-data; name="image"; filename="image.jpg"\r\n'
    content += b'Content-Type: image/jpeg\r\n\r\n'
    content += buffer.getvalue() + b'\r\n'

    # Add the closing boundary to the multipart content
    content += b'--boundary--\r\n'
  
  return content

def images_to_b64(images : list[Image.Image]) -> str:
  """Convert a list of images into a JSON string with each image encoded in base64 format"""

  # * Initialize an empty list to store the dictionaries that afterwords will be converted to a JSON:
  encoded_images_list = []
  # * Index each image with its respective index
  for num_image, image in enumerate(images):
    # * Create a memory buffer (Space of memory in RAM) to store the image data in bytes
    buffer = BytesIO()
    # * Save each image in memory
    image.save(buffer, format="PNG")
    # * Encode each buffer's value (Image in binary) in base64
      # * Buffer.getvalue() gets the image binary data in hex format --> xd9\x80\xb4\xa1\x14I^ \xd3\x94\xd8$\x11\xac\x14\x82\xc2\xe3\x9a\xc1\x80\xe8\x01d\x920\'\x16\x13\x91\x1bJ}\xebI\xa5\xc0\xe3|r\xc3:Y
      # * Then encode this binary data in base64:
        # * Example: "Python"
        # * - Take the ASCII value of each character in the string --> P, y, t, h, o, n are 15, 50, 45, 33, 40, 39
        # * - Calculate the 8-bit binary equivalent of the ASCII values --> 15, 50, 45, 33, 40, 39 are 01010000 01111001 01110100 01101000 01101111 01101110
        # * - Convert the 8-bit chunks into chunks of 6 bits by simply re-grouping the digits --> 01010000 01111001 01110100 01101000 01101111 01101110 are 010100 000111 100101 110100 011010 000110 111101 101110
        # * - Convert the 6-bit binary groups to their respective decimal values. --> 010100 000111 100101 110100 011010 000110 111101 101110 are 20, 7, 37, 52, 26, 6, 61, 46
        # * - Using a base64 encoding table, assign the respective base64 character for each decimal value --> 20, 7, 37, 52, 26, 6, 61, 46 are UHl0aG9u
    encodedB64_image = base64.b64encode(buffer.getvalue())
    # * Decode it in ascii in order to be a string compatible with JSON
    encodedB64_image_string = encodedB64_image.decode('ascii')
    # * Create each instance of the dictionary with the following values:
      # * Image number
      # * Encoded image string
    encoded_image_dict = {
            "Image number": num_image + 1,
            "Encoded image": encodedB64_image_string,
            }
    encoded_images_list.append(encoded_image_dict)
  # * Convert the image dictionary to a JSON and return it
  jsonImages = json.dumps(encoded_images_list)
  return jsonImages

def images_to_b64_v2(images : list[Image.Image]) -> str:
  """Convert a list of images into a JSON string with each image encoded in base64 format"""

  # * Initialize an empty list to store the base 64 images strings
  encoded_images_list = []
  # * Index each image with its respective index
  for image in images:
    # * Create a memory buffer (Space of memory in RAM) to store the image data in bytes
    buffer = BytesIO()
    # * Save each image in memory
    image.save(buffer, format="PNG")
    # * Encode each buffer's value (Image in binary) in base64
      # * Buffer.getvalue() gets the image binary data in hex format --> xd9\x80\xb4\xa1\x14I^ \xd3\x94\xd8$\x11\xac\x14\x82\xc2\xe3\x9a\xc1\x80\xe8\x01d\x920\'\x16\x13\x91\x1bJ}\xebI\xa5\xc0\xe3|r\xc3:Y
      # * Then encode this binary data in base64:
        # * Example: "Python"
        # * - Take the ASCII value of each character in the string --> P, y, t, h, o, n are 15, 50, 45, 33, 40, 39
        # * - Calculate the 8-bit binary equivalent of the ASCII values --> 15, 50, 45, 33, 40, 39 are 01010000 01111001 01110100 01101000 01101111 01101110
        # * - Convert the 8-bit chunks into chunks of 6 bits by simply re-grouping the digits --> 01010000 01111001 01110100 01101000 01101111 01101110 are 010100 000111 100101 110100 011010 000110 111101 101110
        # * - Convert the 6-bit binary groups to their respective decimal values. --> 010100 000111 100101 110100 011010 000110 111101 101110 are 20, 7, 37, 52, 26, 6, 61, 46
        # * - Using a base64 encoding table, assign the respective base64 character for each decimal value --> 20, 7, 37, 52, 26, 6, 61, 46 are UHl0aG9u
    encodedB64_image = base64.b64encode(buffer.getvalue())
    encoded_images_list.append(encodedB64_image)
  return encoded_images_list

def size_upload_files(file):
  """Get the size of an UploadFile file and return it"""

  # * Get the file size (in bytes)
  file.file.seek(0, 2)
  file_size = file.file.tell()

  # * Move the cursor back to the beginning
  file.seek(0)

  return file_size