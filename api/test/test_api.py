from fastapi.testclient import TestClient
from api.api import app

client = TestClient(app=app)