#https://itsjoshcampos.codes/fast-api-api-key-authorization
#https://joshdimella.com/blog/adding-api-key-auth-to-fast-api
#https://testdriven.io/tips/6840e037-4b8f-4354-a9af-6863fb1c69eb/

from dotenv import load_dotenv, find_dotenv

import os

from fastapi.security.api_key import APIKeyHeader, APIKeyQuery, APIKeyCookie
from fastapi import Security, HTTPException, Depends
from starlette.status import HTTP_401_UNAUTHORIZED

# * Find the path of the dotenv file
dotenv_path = find_dotenv(filename='api/auth/.env', raise_error_if_not_found=True)

# * Load the enviromental variables from the dotenv path
load_dotenv(dotenv_path)

# * Load the list of API KEYS
API_KEYS = os.getenv(key='API_KEYS')

API_KEY_NAME = 'accces-token-api-key'

# * Create instances of api key methods
api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_cookie = APIKeyCookie(name=API_KEY_NAME, auto_error=False)

async def get_api_key(
    api_key_query: str = Security(api_key_query),
    api_key_header: str = Security(api_key_header),
    api_key_cookie: str = Security(api_key_cookie),) -> str:

    """Retrieve and validate an API key from the query parameters the HTTP header or via the cookies.

    Args:
        api_key_query: The API key passed as a query parameter.
        api_key_header: The API key passed in the HTTP header.
        api_key_cookie: The API key passed via the cookies.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If the API key is invalid or missing.
    """



    if (api_key_query == None and api_key_header == None and api_key_cookie == None):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
        )
    
    if api_key_query in API_KEYS:
        return api_key_query
    elif api_key_header in API_KEYS:
        return api_key_header
    elif api_key_cookie in API_KEYS:
        return api_key_cookie
    else:
        raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )