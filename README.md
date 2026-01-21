# Optima: Business Text Classification System

Optima is a dual-model text classification system designed to categorize customer support queries into **Billing**, **Technical**, or **General** categories. It leverages a hybrid approach using **LinearSVC** for high-efficiency classification and **DistilBERT** for deep contextual understanding.

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
![b553ad32-fc93-4074-bda9-b26e8373b913](https://github.com/user-attachments/assets/17caf017-b602-44a4-b49e-a85ca96daa1b)

![856bb12c-8632-45ca-9197-5375ab189da0](https://github.com/user-attachments/assets/28ee13ed-a5c3-47cc-8576-8c10a5e6b956)
![0f89d923-315f-4411-851b-2c9af9aabfd5](https://github.com/user-attachments/assets/bdf576d3-f67f-413c-9ce5-ac28c57829d9)

![5ad8f7bb-22e4-4082-9396-ac74d8f52514](https://github.com/user-attachments/assets/7f604ebe-fab8-42be-af02-92604888ba79)
