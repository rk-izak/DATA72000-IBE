import os
import json
import getpass


def set_api_key(file_path: str = "./utils/keys.json", api_name: str = "") -> None:
    """
    Set the API key as an environment variable based on the provided API name.

    Args:
        file_path (str): Path to the JSON file containing API keys.
        api_name (str): Name of the API to set the key for.

    Returns:
        None
    """
    try:
        with open(file_path, 'r') as file:
            api_keys = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{file_path}'.")
        return

    api_dict = {api["name"].lower(): api["key"] for api in api_keys}
    api_name = api_name.lower()

    if api_name in api_dict:
        env_var_name = f"{api_name.upper()}_API_KEY"
        os.environ[env_var_name] = api_dict[api_name]

        if api_name == "langchain":
            os.environ["LANGCHAIN_TRACING_V2"] = "true"

        if not os.environ.get(env_var_name):
            os.environ[env_var_name] = getpass.getpass(f"{env_var_name}: ")
    else:
        print(f"API name '{api_name}' not found in the provided file.")

