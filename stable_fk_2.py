import os
import requests
import json
import base64

from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify

# Find the path of the dotenv file
dotenv_path = find_dotenv()

# Load the environmental variables from the dotenv path
load_dotenv(dotenv_path)

# Endpoint for getting all the current Stable Diffusion Engines
# TODO: Change api_host to host it locally
api_host = os.getenv('API_HOST', 'https://api.stability.ai')
endpoint_engines = f"{api_host}/v1/engines/list"

# Getting the API key and checking if it actually exists
api_key = os.getenv("STABILITY_API_KEY")
if api_key is None:
    raise Exception("Missing Stability API key.")

app = Flask(__name__)

@app.route("/get-engines")
def get_engines():
    engines = fetch_engines()
    return jsonify(engines)

@app.route("/create-image", methods=["POST"])
def create_image():
    prompt = request.json.get("prompt")
    engine_id = request.json.get("engine_id")
    if prompt is None or engine_id is None:
        return jsonify({"error": "Invalid request. Missing prompt or engine_id."}), 400
    
    try:
        image_data = generate_image(prompt, engine_id)
        return jsonify({"image_data": image_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def fetch_engines():
    response = requests.get(endpoint_engines, headers={"Authorization": f"Bearer {api_key}"})
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    engines = response.json()
    return engines

def generate_image(prompt, engine_id):
    endpoint_text_to_image = f"{api_host}/v1/generation/{engine_id}/text-to-image"
    response = requests.post(
        endpoint_text_to_image,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": prompt
                }
            ],
            "cfg_scale": 7,
            "clip_guidance_preset": "FAST_BLUE",
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 30,
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    image_data = []
    for i, image in enumerate(data["artifacts"]):
        image_data.append(image["base64"])

    return image_data

if __name__ == "__main__":
    app.run(host="localhost", port=5000)