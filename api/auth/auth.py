#https://itsjoshcampos.codes/fast-api-api-key-authorization
#https://joshdimella.com/blog/adding-api-key-auth-to-fast-api
#https://testdriven.io/tips/6840e037-4b8f-4354-a9af-6863fb1c69eb/

from dotenv import load_dotenv, find_dotenv

import os

from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from fastapi import Security, HTTPException, Depends
from starlette.status import HTTP_401_UNAUTHORIZED

# * Find the path of the dotenv file
dotenv_path = find_dotenv(filename='api/auth/.env', raise_error_if_not_found=True)

# * Load the enviromental variables from the dotenv path
load_dotenv(dotenv_path)

# * Load the list of API KEYS
API_KEYS = os.getenv(key='API_KEYS')

api_key_query = APIKeyQuery(name="api-key", auto_error=False)
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def get_api_key(
    api_key_query: str = Security(api_key_query),
    api_key_header: str = Security(api_key_header),
) -> str:
    """Retrieve and validate an API key from the query parameters or HTTP header.

    Args:
        api_key_query: The API key passed as a query parameter.
        api_key_header: The API key passed in the HTTP header.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If the API key is invalid or missing.
    """
    if api_key_query in API_KEYS:
        return api_key_query
    if api_key_header in API_KEYS:
        return api_key_header
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )