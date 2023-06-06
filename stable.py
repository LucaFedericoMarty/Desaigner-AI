import os
import requests
import json

from dotenv import load_dotenv, find_dotenv 

# * Find the path of the dotenv file

dotenv_path = find_dotenv()

# * Load the enviromental variables from the dotenv path

load_dotenv(dotenv_path)

# * Endpoint for getting all the current Stable Diffusion Engines

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

    return engines
