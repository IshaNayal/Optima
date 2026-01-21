import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os

def train_model():
    # Load data
    try:
        df = pd.read_csv('data/dataset.csv')
    except FileNotFoundError:
        print("Dataset not found. Please run generate_data.py first.")
        return

    X = df['text']
    y = df['category']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline with enhanced n-grams and SVM
    # LinearSVC is generally better for text classification than LogisticRegression
    # CalibratedClassifierCV is used to get probability estimates (predict_proba)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=10000)),
        ('clf', CalibratedClassifierCV(LinearSVC(random_state=42, C=1.0, max_iter=2000), cv=5))
    ])

    # Train
    print("Training model (LinearSVC with Calibration)...")
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save metrics to JSON for frontend display
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics_data = {
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    import json
    with open('models/metrics_linear.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print("Metrics saved to models/metrics_linear.json")

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/text_classifier.joblib')
    print("\nModel saved to models/text_classifier.joblib")

if __name__ == "__main__":
    train_model()
