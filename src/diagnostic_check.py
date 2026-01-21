import os
import sys
import importlib.util


def check_import(module_name):
    if importlib.util.find_spec(module_name) is None:
        print(f"[-] {module_name}: NOT FOUND")
        return False
    else:
        print(f"[+] {module_name}: INSTALLED")
        return True


def check_file(path):
    if os.path.exists(path):
        print(f"[+] File found: {path}")
        return True
    else:
        print(f"[-] File MISSING: {path}")
        return False


def run_diagnostics():
    print("==================================================")
    print("       OPTIMA SYSTEM DIAGNOSTICS CHECK            ")
    print("==================================================")
    
   
    print("\n[1] Checking Core Dependencies...")
    dependencies = ['torch', 'transformers', 'sklearn', 'fastapi', 'uvicorn', 'streamlit', 'pandas']
    all_deps_ok = all(check_import(dep) for dep in dependencies)
    
    
    print("\n[2] Checking Model Artifacts...")
    models = [
        "models/linear_svc_model.joblib",
        "models/bert/config.json",
        "models/bert/model.safetensors" 
        
    ]
    
    bert_dir = "models/bert"
    linear_model = "models/text_classifier.joblib"
    
    models_ok = check_file(linear_model)
    if os.path.isdir(bert_dir):
        print(f"[+] BERT Directory found: {bert_dir}")
        
        if os.path.exists(os.path.join(bert_dir, "model.safetensors")) or os.path.exists(os.path.join(bert_dir, "pytorch_model.bin")):
             print(f"[+] BERT Weights found in {bert_dir}")
        else:
             print(f"[-] BERT Weights MISSING in {bert_dir}")
             models_ok = False
    else:
        print(f"[-] BERT Directory MISSING: {bert_dir}")
        models_ok = False

    
    print("\n[3] Checking Datasets...")
    datasets = ["data/dataset.csv", "data/bitext_raw.csv"]
    data_ok = all(check_file(f) for f in datasets)

    print("\n==================================================")
    if all_deps_ok and models_ok and data_ok:
        print("RESULT: SYSTEM HEALTHY ✅")
        print("Ready to run: uvicorn src.app:app --reload")
    else:
        print("RESULT: SYSTEM ISSUES DETECTED ❌")
        print("Please resolve missing dependencies or files.")
    print("==================================================")


if __name__ == "__main__":
    run_diagnostics()
