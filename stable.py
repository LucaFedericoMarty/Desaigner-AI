import os
import requests
import json
import base64

from dotenv import load_dotenv, find_dotenv 

# * Find the path of the dotenv file

dotenv_path = find_dotenv()

# * Load the enviromental variables from the dotenv path

load_dotenv(dotenv_path)

# * Endpoint for getting all the current Stable Diffusion Engines

# TODO: Change api_host to host it locally

api_host = os.getenv('API_HOST', 'https://api.stability.ai')

endpoint_engines = f"{api_host}/v1/engines/list"

# * Getting the API key and checking if it actually exists

api_key = os.getenv("STABILITY_API_KEY")
if api_key is None:
    raise Exception("Missing Stability API key.")

def get_engines(endpoint, api_key):

# * Do the request with the following parameters:
# * METHOD: Get
# * ENDPOINT: /v1/engines/list
# * HEADER: Stable Diffusion API Key

    response = requests.get(endpoint, headers={"Authorization": f"Bearer {api_key}"})
    
    # * In case the status code is not 200 (that means everything went correct), raise an error

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    # * Receive the engines JSON
    engines = response.json()

    # * Create a new JSON file and write the engines to it
    # * Then, close the file

    jsonString = json.dumps(engines)
    enginesJSON = open("engines.json", "w")
    enginesJSON.write(jsonString)
    enginesJSON.close()

    return jsonString

def filter_engine(engines, engine_key_word):

    # * Parse JSON file to Python dictionary

    engines_dict = json.loads(engines)

    # * Index all the engines
    # * Search for the engine which has the key word
    # * Once found, grab its id

    filtered_engine_id = [engine["id"] for engine in engines_dict if engine_key_word in engine["description"]][0]

    return filtered_engine_id 

def create_image(endpoint, api_key, prompt):
    response = requests.post(
    endpoint,
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

    for i, image in enumerate(data["artifacts"]):
        with open(f"./images/image {i}.png", "wb") as f:
            f.write(base64.b64decode(image["base64"]))

def create_prompt(price, style, enviroment):
    return []

engines = get_engines(endpoint_engines, api_key)
engine_id = filter_engine(engines, "XL")
print(engine_id)

endpoint_text_to_image = f"{api_host}/v1/generation/{engine_id}/text-to-image"

# * TEST GIT

create_image(endpoint_text_to_image, api_key, "Gaming room with fancy lights and a pool table")

