import requests
import json

def check_api():
    url = "http://localhost:8000/predict"
    text = "My server is broken"
    
    try:
        response = requests.post(url, json={"text": text})
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_api()
