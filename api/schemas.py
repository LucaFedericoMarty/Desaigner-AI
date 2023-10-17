from fastapi import UploadFile, Query, Security, File, Response
from fastapi.security.api_key import APIKey

from pydantic import BaseModel, Field
from typing import Annotated, Optional, List, Dict

from PIL import Image

from api.auth.auth import get_api_key

IMAGES_B64 = List[str]

IMAGE = Image.Image

#Field(default=None, title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty")

class BaseParams(BaseModel):
    budget : str = Field(title="Budget of the re-design",description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs", examples=["Low", "Medium", "High"])
    style : str = Field(title="Style of the re-design", description="Choose any interior design style that best suits your desires", examples=["Modern", "Minimalist", "Rustic", "Art Deco", "Traditional", "Classic"])
    environment : str = Field(title="Environment of the re-design", description="The environment you are looking to re-design", examples=["Living room", "Dinning room", "Bathroom", "Bedroom"])
    weather : str  = Field(title="Weather of the region", description="The typical weather you of the region you are living", examples=["Hot", "Tropical", "Rainy", "Snowy"])
    disability : Optional[str] = Field(None, title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty")
    steps : int = Field(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference",default=20, ge=1, le=50)
    guidance_scale : float = Field(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", default=7, ge=3.5 , le=7.5) 
    num_images : int = Field(title="Number of images to create", description="The higher the number, the more time required to create the images" , default=2, ge=1, le=4)

class Txt2ImgParams(BaseParams):
    """Pydantic model for Text2Image operations
    
    It has all the necessary parameters for performing a Text2Image request. The attributes are the following:

    - Budget: Budget of the re-design
    - Style: Style of the re-design
    - Environment: Environment of the re-design
    - Weather: Weather of the region
    - Disability: Type of disability of the user
    - Steps: Number of steps necessary to create images
    - Guidance scale: Number that represents the fidelity of prompt when creating the image
    - Num_images: Number of images to create
    """

    pass

class Img2ImgParams(BaseParams):
    """Pydantic model for Image2Image operations
    
    It has all the necessary parameters for performing a Image2Image request. The attributes are the following:

    - Budget: Budget of the re-design
    - Style: Style of the re-design
    - Environment: Environment of the re-design
    - Weather: Weather of the region
    - Disability: Type of disability of the user
    - Steps: Number of steps necessary to create images
    - Guidance scale: Number that represents the fidelity of prompt when creating the image
    - ControlNet Conditioning Scale: Number that represents the fidelity of the image in the final result
    - Num_images: Number of images to create
    """
    controlnet_conditioning_scale : float = Field(title="Number that represents the fidelity of the image in the final result", description="Higher conditioning scale tends to generate images that are more similar to the input image", default=1, ge=0.5, le=1)

class InpaintParams(BaseParams):
    """Pydantic model for Inpaint operations
    
    It has all the necessary parameters for performing a Inpaint request. The attributes are the following:

    - Budget: Budget of the re-design
    - Style: Style of the re-design
    - Environment: Environment of the re-design
    - Weather: Weather of the region
    - Disability: Type of disability of the user
    - Steps: Number of steps necessary to create images
    - Guidance scale: Number that represents the fidelity of prompt when creating the image
    - Num_images: Number of images to create
    """

    pass

class ImageResponse(BaseModel):
    """Pydantic model for image responses
    
    Attributes:
    
    - Images: List of images in base64 format"""
    images: IMAGES_B64 = Field(..., description="List of images in base64 format")