import requests
import sys

def test_api():
    url = "http://localhost:8000/predict"
    
    test_cases = [
        "The app freezes whenever I try to download my billing statement."
    ]
    
    print("Testing API...")
    for text in test_cases:
        try:
            response = requests.post(url, json={"text": text})
            if response.status_code == 200:
                result = response.json()
                print(f"Text: '{text}' -> Category: {result['category']} (Confidence: {result['confidence']:.2f})")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            print("Could not connect to the API. Is it running?")
            sys.exit(1)

if __name__ == "__main__":
    test_api()
