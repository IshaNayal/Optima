from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import torch.nn.functional as F
import joblib
from typing import Dict

app = FastAPI(title="Text Classification API (BERT + LinearSVC)")


BERT_MODEL_DIR = 'models/bert'
bert_model = None
bert_tokenizer = None

if os.path.exists(BERT_MODEL_DIR):
    print("Loading BERT model...")
    bert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_DIR)
    bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    bert_model.eval()
else:
    print("BERT model not found. API will fail for BERT requests.")


SKLEARN_MODEL_PATH = 'models/text_classifier.joblib'
sklearn_model = None

if os.path.exists(SKLEARN_MODEL_PATH):
    print("Loading LinearSVC model...")
    sklearn_model = joblib.load(SKLEARN_MODEL_PATH)
else:
    print("LinearSVC model not found.")


label_map = {0: 'Billing', 1: 'General', 2: 'Technical'}

class PredictRequest(BaseModel):
    text: str
    model: str = "bert" 

class PredictResponse(BaseModel):
    category: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if request.model == "sklearn":
        if not sklearn_model:
            raise HTTPException(status_code=500, detail="LinearSVC model not loaded.")
        try:
            prediction = sklearn_model.predict([request.text])[0]
            probs = sklearn_model.predict_proba([request.text])[0]
            classes = sklearn_model.classes_
            prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
            confidence = prob_dict[prediction]
            
            return PredictResponse(
                category=prediction,
                confidence=confidence,
                probabilities=prob_dict,
                model_used="LinearSVC"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    else: 
        if not bert_model:
            raise HTTPException(status_code=500, detail="BERT model not loaded.")
        try:
            inputs = bert_tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=64)
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).numpy()[0]
                predicted_class_id = torch.argmax(logits, dim=1).item()
            
            predicted_label = label_map[predicted_class_id]
            prob_dict = {label_map[i]: float(probs[i]) for i in range(len(label_map))}
            
            return PredictResponse(
                category=predicted_label, 
                confidence=float(probs[predicted_class_id]),
                probabilities=prob_dict,
                model_used="BERT"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Text Classification API is running. Supports 'bert' and 'sklearn' models."}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

