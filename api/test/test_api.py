from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK, HTTP_413_REQUEST_ENTITY_TOO_LARGE, HTTP_415_UNSUPPORTED_MEDIA_TYPE
from api.api import app
from schemas import IMAGES_B64

client = TestClient(app=app)

request_data = {
    "budget": "High", 
    "style": "Modern",
    "environment": "Living room",
    "weather": "Normal",
    "disability": "x",
    "steps": 20,
    "guidance_scale": 7,
    "num_images": 2}

def test_text_to_image():
    """Test for text to image endpoint
    
    Examines:

    - Status code: 200
    - Response type: IMAGES_B64 (list of images in base64 encoded)

    """
    response = client.post('/txt2img/v2/v1', json=request_data)
    assert response.status_code == HTTP_200_OK, f"Expected '{HTTP_200_OK}' but obtained '{response.status_code}"
    assert type(response.content) == IMAGES_B64, f"Expected '{IMAGES_B64}' but obtained '{type(response.content)}"