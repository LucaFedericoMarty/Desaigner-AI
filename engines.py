import os
import requests

import dotenv

# * Endpoint for getting all the current Stable Diffusion Engines

api_host = os.getenv('API_HOST', 'https://api.stability.ai')
url = f"{api_host}/v1/engines/list"

# * Getting the API key and checking if it actually exists

api_key = os.getenv("STABILITY_API_KEY")
if api_key is None:
    raise Exception("Missing Stability API key.")

# * Do the  

response = requests.get(url, headers={
    "Authorization": f"Bearer {api_key}"
})

if response.status_code != 200:
    raise Exception("Non-200 response: " + str(response.text))

# Do something with the payload...
payload = response.json()