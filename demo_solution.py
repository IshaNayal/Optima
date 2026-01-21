import requests

API_URL = "http://localhost:8000/predict"

def test_model(model_name, text):
    print(f"\n--- Testing {model_name} ---")
    payload = {"text": text, "model": model_name}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        print(f"Input: '{text}'")
        print(f"Predicted Category: {data['category']}")
        print(f"Confidence: {data['confidence']:.4f}")
        print(f"Model Used: {data['model_used']}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("DEMONSTRATING SOLUTION COMPONENTS")
    print("=================================")
    
    # 1. Model Usage via API
    text = "I need a copy of my last invoice please."
    test_model("sklearn", text)
    test_model("bert", text)
    
    print("\n=================================")
    print("EVALUATION METRICS LOCATIONS")
    print("1. LinearSVC Metrics: models/metrics_linear.json")
    print("2. BERT Metrics:      models/metrics_bert.json")
    print("3. Visual Dashboard:  http://localhost:8502 (See 'Model Evaluation' tab)")
