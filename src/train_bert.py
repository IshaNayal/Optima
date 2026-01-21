import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os


torch.manual_seed(42)
np.random.seed(42)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'macro_f1': f1,
        'macro_precision': precision,
        'macro_recall': recall
    }

def get_bitext_mapping(intent):
    intent_mapping = {
        
        'check_invoice': 'Billing', 'get_invoice': 'Billing',
        'check_payment_methods': 'Billing', 'payment_issue': 'Billing',
        'check_refund_policy': 'Billing', 'get_refund': 'Billing', 'track_refund': 'Billing',
        'check_cancellation_fee': 'Billing',
        
        
        'registration_problems': 'Technical',
        
        
        'cancel_order': 'General', 'change_order': 'General', 'place_order': 'General', 'track_order': 'General',
        'change_shipping_address': 'General', 'set_up_shipping_address': 'General', 
        'delivery_options': 'General', 'delivery_period': 'General',
        'complaint': 'General', 'review': 'General', 
        'contact_customer_service': 'General', 'contact_human_agent': 'General',
        'create_account': 'General', 'delete_account': 'General', 'edit_account': 'General', 
        'switch_account': 'General', 'recover_password': 'General',
        'newsletter_subscription': 'General'
    }
    return intent_mapping.get(intent, None)

def train_bert():
    print("=== BERT Training Start ===")
    
    
    print("\n[Phase 1] Loading IDE-agent dataset (dataset.csv)...")
    try:
        df_ide = pd.read_csv('data/dataset.csv')
    except FileNotFoundError:
        print("IDE dataset not found.")
        return

    label_map = {'Billing': 0, 'General': 1, 'Technical': 2}
    df_ide['label'] = df_ide['category'].map(label_map)
    df_ide = df_ide.dropna(subset=['text', 'label'])
    df_ide['label'] = df_ide['label'].astype(int)

    
    print("Using 4000 samples from IDE dataset for Phase 1 training...")
    df_ide_subset = df_ide.sample(4000, random_state=42)
    
    X_train_ide, X_test_ide, y_train_ide, y_test_ide = train_test_split(
        df_ide_subset['text'], df_ide_subset['label'], test_size=0.2, random_state=42
    )

    train_dataset_ide = Dataset.from_dict({'text': X_train_ide, 'label': y_train_ide})
    test_dataset_ide = Dataset.from_dict({'text': X_test_ide, 'label': y_test_ide})

    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    print("Tokenizing Phase 1 data...")
    tokenized_train_ide = train_dataset_ide.map(tokenize_function, batched=True)
    tokenized_test_ide = test_dataset_ide.map(tokenize_function, batched=True)

    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args_phase1 = TrainingArguments(
        output_dir='./results/phase1',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs/phase1',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        use_cpu=True
    )

    trainer_phase1 = Trainer(
        model=model,
        args=training_args_phase1,
        train_dataset=tokenized_train_ide,
        eval_dataset=tokenized_test_ide,
        compute_metrics=compute_metrics
    )

    print("Starting Phase 1 Training (IDE Data)...")
    trainer_phase1.train() 
    
    print("Evaluating Phase 1 Model on IDE Test Set...")
    metrics_ide = trainer_phase1.evaluate()
    print(f"Phase 1 Metrics (IDE Test Set): {metrics_ide}")
    
    
    print("\n[Diagnostic] Phase 1 Classification Report:")
    preds_phase1 = trainer_phase1.predict(tokenized_test_ide)
    y_preds_1 = np.argmax(preds_phase1.predictions, axis=1)
    print(classification_report(preds_phase1.label_ids, y_preds_1, target_names=['Billing', 'General', 'Technical']))

    
    print("Saving Phase 1 model to models/bert_phase1...")
    os.makedirs('models/bert_phase1', exist_ok=True)
    model.save_pretrained("models/bert_phase1")
    tokenizer.save_pretrained("models/bert_phase1")

    
    print("\n[Phase 2] Loading and Processing HuggingFace dataset (bitext_raw.csv)...")
    try:
        df_hf = pd.read_csv('data/bitext_raw.csv')
        
        df_hf['mapped_category'] = df_hf['intent'].apply(get_bitext_mapping)
        df_hf = df_hf.dropna(subset=['mapped_category']) 
        
        
        df_hf['label'] = df_hf['mapped_category'].map(label_map)
        df_hf = df_hf.dropna(subset=['instruction', 'label'])
        df_hf['label'] = df_hf['label'].astype(int)
        
        
        print("Using 1000 samples from HF dataset for Phase 2 fine-tuning...")
        df_hf_subset = df_hf.sample(1000, random_state=42)
        
        X_train_hf, X_test_hf, y_train_hf, y_test_hf = train_test_split(
            df_hf_subset['instruction'], df_hf_subset['label'], test_size=0.2, random_state=42
        )
        
        train_dataset_hf = Dataset.from_dict({'text': X_train_hf, 'label': y_train_hf})
        test_dataset_hf = Dataset.from_dict({'text': X_test_hf, 'label': y_test_hf})
        
        print("Tokenizing Phase 2 data...")
        tokenized_train_hf = train_dataset_hf.map(tokenize_function, batched=True)
        tokenized_test_hf = test_dataset_hf.map(tokenize_function, batched=True)
        
        
        training_args_phase2 = TrainingArguments(
            output_dir='./results/phase2',
            num_train_epochs=1,
            learning_rate=2e-5, 
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_dir='./logs/phase2',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no", 
            use_cpu=True
        )
        
      
        trainer_phase2 = Trainer(
            model=model,
            args=training_args_phase2,
            train_dataset=tokenized_train_hf,
            eval_dataset=tokenized_test_hf,
            compute_metrics=compute_metrics
        )
        
        print("Starting Phase 2 Training (HF Data - Light Fine-tuning)...")
        trainer_phase2.train()
        
        print("Evaluating Phase 2 Model on HF Test Set...")
        metrics_hf = trainer_phase2.evaluate()
        print(f"Phase 2 Metrics (HF Test Set): {metrics_hf}")
        
        
        print("\n[Diagnostic] Phase 2 Classification Report (HF Test Set):")
        preds_phase2 = trainer_phase2.predict(tokenized_test_hf)
        y_preds_2 = np.argmax(preds_phase2.predictions, axis=1)
        print(classification_report(preds_phase2.label_ids, y_preds_2, target_names=['Billing', 'General', 'Technical']))

        
        print("Re-evaluating Phase 2 Model on IDE Test Set (Stability Check)...")
        metrics_ide_final = trainer_phase2.evaluate(tokenized_test_ide)
        print(f"Phase 2 Metrics on IDE Test Set: {metrics_ide_final}")
        
        
        acc_phase1 = metrics_ide['eval_accuracy']
        acc_phase2 = metrics_ide_final['eval_accuracy']
        
        if acc_phase2 < (acc_phase1 - 0.05): 
            print(f"\n[WARNING] Phase 2 degraded performance on PRIMARY dataset (Phase 1 Acc: {acc_phase1:.4f} -> Phase 2 Acc: {acc_phase2:.4f}).")
            print("Reverting to Phase 1 model as per instruction 'Do NOT prioritize noisy data over clean data'.")
            
           
            model = DistilBertForSequenceClassification.from_pretrained("models/bert_phase1")
            tokenizer = DistilBertTokenizer.from_pretrained("models/bert_phase1")
        else:
            print("\nPhase 2 fine-tuning successful and stable.")

    except Exception as e:
        print(f"Skipping Phase 2 due to error or missing data: {e}")

    
    print("\nSaving final model to models/bert...")
    os.makedirs('models/bert', exist_ok=True)
    model.save_pretrained("models/bert")
    tokenizer.save_pretrained("models/bert")
    print("BERT model saved to models/bert")

if __name__ == "__main__":
    train_bert()
