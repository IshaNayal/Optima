import requests
import pandas as pd
import io
import os

def fetch_data():
    url = "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    print(f"Downloading from {url}...")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Download successful. Parsing CSV...")
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            print("CSV parsed successfully.")
            
            print("\nColumns:", df.columns.tolist())
            print("\nFirst 5 rows:")
            print(df.head())
            
            # Save locally
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/bitext_raw.csv', index=False)
            print("\nSaved raw data to data/bitext_raw.csv")
            
            # Inspect unique intents
            if 'intent' in df.columns:
                print("\nUnique Intents:")
                print(df['intent'].unique())
            elif 'category' in df.columns:
                 print("\nUnique Categories:")
                 print(df['category'].unique())
                 
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            print("Response:", response.text[:200])
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_data()
