# Optima
Smart Text Classifier
 Business Text Classification System

This is a dual-model text classification system designed to categorize customer support queries into **Billing**, **Technical**, or **General** categories. It leverages a hybrid approach using **LinearSVC** for high-efficiency classification and **DistilBERT** for deep contextual understanding.

## Features
- **Dual-Model Architecture**: 
  - **LinearSVC**: Fast, interpretable, CPU-efficient (F1: ~0.998).
  - **DistilBERT**: Context-aware, robust for ambiguous queries.
- **API**: FastAPI-based REST API supporting model selection (`sklearn` vs `bert`).
- **Frontend**: Streamlit-based interactive UI with real-time predictions and confidence visualization.
- **Evaluation**: Detailed metrics tracking (Confusion Matrix, Precision/Recall).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/IshaNayal/Optima.git
   cd Optima
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure `torch` is installed compatible with your system)*

## Usage

### 1. Run the API
```bash
uvicorn src.app:app --reload
```

### 2. Run the Frontend
```bash
streamlit run src/streamlit_app.py
```

## Project Structure
- `src/`: Source code for API and training.
- `models/`: Trained models (LinearSVC) and metrics. *Note: BERT weights are excluded due to size.*
- `data/`: Datasets used for training.

## Training
To retrain the models:
```bash
python src/train.py       # Train LinearSVC
python src/train_bert.py  # Train BERT
```
