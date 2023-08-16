from fastapi.testclient import TestClient
from api.api import app

client = TestClient(app=app)

def test_text_to_image():
    response = client.post('/txt2img/v2/v1', json={"budget": "Low", "style": "Oceanic","environment": "Bedroom","weather": "Tropical","disability": "x","steps": 20,"guidance_scale": 7,"num_images": 2})