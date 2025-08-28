import requests
import json

url = "http://localhost:5000/forecast"
headers = {"Content-Type": "application/json"}
payload = {"steps": 11}

try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()  # Raise an exception for bad status codes
    print("Success! Response from API:")
    print(json.dumps(response.json(), indent=4))
except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")