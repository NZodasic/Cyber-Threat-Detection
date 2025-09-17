# test_model.py
import joblib
import numpy as np
import os
import argparse

MAX_BYTES = 2000  # phải giống với khi bạn build dataset / train

def extract_byte_features(filepath, max_len=MAX_BYTES):
    """Đọc file .exe và trả về vector (1, max_len) padded/truncated"""
    with open(filepath, "rb") as f:
        data = f.read(max_len)
    arr = list(data)
    if len(arr) < max_len:
        arr += [0] * (max_len - len(arr))
    return np.array(arr, dtype=np.int32).reshape(1, -1)

def load_model(model_path="malware_detector.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    return model

def predict_file(model, filepath):
    X = extract_byte_features(filepath)
    pred = model.predict(X)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
    return pred, prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test malware model on one .exe file")
    parser.add_argument("exe_path", help="Path to the .exe file to test")
    parser.add_argument("--model", default="malware_detector.pkl", help="Path to saved .pkl model")
    args = parser.parse_args()

    model = load_model(args.model)
    if not os.path.exists(args.exe_path):
        raise FileNotFoundError(f"Exe file not found: {args.exe_path}")

    label, prob = predict_file(model, args.exe_path)
    print("Prediction:", label)
    if prob is not None:
        # If binary, prob may be [p_benign, p_virus] or vice versa depending on training labels
        print("Probabilities:", prob)
    else:
        print("Model does not provide predict_proba output.")
