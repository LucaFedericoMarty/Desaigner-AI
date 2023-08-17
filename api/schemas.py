from fastapi import UploadFile, Query, Security, File
from fastapi.security.api_key import APIKey

from pydantic import BaseModel, Field
from typing import Annotated, Optional, List, Dict

from api.auth.auth import get_api_key

IMAGES_B64 = List[str]

class Txt2ImgParamas(BaseModel):
    budget : str = Field(title="Budget of the re-design",description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs", examples=["Low", "Medium", "High"])
    style : str = Field(title="Style of the re-design", description="Choose any interior design style that best suits your desires", examples=["Modern", "Minimalist", "Rustic", "Art Deco", "Traditional", "Classic"])
    environment : str = Field(title="Environment of the re-design", description="The environment you are looking to re-design", examples=["Living room", "Dinning room", "Bathroom", "Bedroom"])
    weather : str  = Field(title="Weather of the region", description="The typical weather you of the region you are living", examples=["Hot", "Tropical", "Rainy", "Snowy"])
    disability : str | None or None = Field(title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty", examples=["Blindness", "Deaf"])
    steps : int = Field(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference",default=20, ge=15, le=50)
    guidance_scale : float = Field(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", default=7, ge=3.5 , le=7.5) 
    num_images : int = Field(title="Number of images to create", description="The higher the number, the more time required to create the images" , default=2, ge=2, le=6)

class Img2ImgParams(BaseModel):
    budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")]
    style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")]
    environment : Annotated[str , Query(title="Environment of the re-design", description="The environment you are looking to re-design")]
    weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")]
    disability : Annotated[str , Query(title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty")] = ""
    input_image : Annotated[UploadFile, File(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image")] 
    steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20 
    guidance_scale : Annotated[float, Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5 
    num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2

class InpaintParams(BaseModel):
    budget : Annotated[str , Query(title="Budget of the re-design", description="Higher budget tends to produce better re-designs, while lower budget tends to produce worse re-designs")]
    style : Annotated[str , Query(title="Style of the re-design", description="Choose any interior design style that best suits your desires")]
    environment : Annotated[str , Query(title="Environment of the re-design", description="The environment you are looking to re-design")]
    weather : Annotated[str , Query(title="Weather of the region", description="The typical weather you of the region you are living")]
    disability : Annotated[str , Query(title="Type of disability of the user", description="In case the user has a disability, the user should enter the disabilty")] = ""
    input_image : Annotated[UploadFile , File(title="Image desired to re-design", description="The model will base the re-design based on the characteristics of this image")]
    mask_image : Annotated[UploadFile , File(title="Image mask of the input image", description="This image should be in black and white, and the white parts should be the parts you want to change and the black parts the ones you want to maintain")]
    steps : Annotated[int , Query(title="Number of steps necessary to create images", description="More denoising steps usually lead to a higher quality image at the expense of slower inference", ge=10, le=50)] = 20
    guidance_scale : Annotated[float , Query(title="Number that represents the fidelity of prompt when creating the image", description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality", ge=3.5 , le=7.5)] = 4.5 
    num_images : Annotated[int , Query(title="Number of images to create", description="The higher the number, the more time required to create the images" , ge=2, le=6)] = 2

class ImageResponse(BaseModel):
    images: IMAGES_B64 = Field(..., description="List of images in base64 format")