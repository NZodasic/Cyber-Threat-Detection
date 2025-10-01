"""
Extract features + predict malware family / benign for a PE file.

Usage:
  python predict.py --file sample.exe --rf
  python predict.py --file sample.exe --mlp
  python predict.py --file sample.exe --debug

Requirements:
  pip install pefile pandas numpy scikit-learn joblib torch scipy
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd

import pefile
import hashlib

def extract_features(path, label=""):
    try:
        pe = pefile.PE(path)
    except pefile.PEFormatError:
        print(f"[!] {path} is not a valid PE file.")
        return None

    features = {}
    # Hash MD5
    with open(path, "rb") as f:
        b = f.read()
    features['md5'] = hashlib.md5(b).hexdigest()

    try:
        features['SizeOfOptionalHeader'] = pe.OPTIONAL_HEADER.SizeOfHeaders
        features['SizeOfCode'] = pe.OPTIONAL_HEADER.SizeOfCode
        features['SizeOfImage'] = pe.OPTIONAL_HEADER.SizeOfImage
        features['NumberOfSections'] = len(pe.sections)
        features['DllCharacteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
    except Exception:
        features.setdefault('SizeOfOptionalHeader', 0)
        features.setdefault('SizeOfCode', 0)
        features.setdefault('SizeOfImage', 0)
        features.setdefault('NumberOfSections', 0)
        features.setdefault('DllCharacteristics', 0)

    # Collect imports
    imports = []
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                if imp.name:
                    try:
                        imports.append(imp.name.decode(errors='ignore'))
                    except Exception:
                        imports.append(str(imp.name))
    imports_text = " ".join(imports)
    features['imports_text'] = imports_text

    features['label'] = label
    return features

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def load_preprocessor(path="preprocessor.joblib"):
    return joblib.load(path)

def build_input_df(feats: dict, preprocessor, debug=False):
    df = pd.DataFrame([feats])
    if 'imports_text' not in df:
        df['imports_text'] = ""
    if debug:
        print("Preview features row:", df.head(1).to_dict(orient="records")[0])
    return df

def predict_with_rf(preprocessor, model, le, df):
    X = preprocessor.transform(df)
    from scipy import sparse
    if sparse.issparse(X):
        X = X.toarray()
    probs = model.predict_proba(X)[0]
    pred_idx = np.argmax(probs)
    pred_class = le.inverse_transform([pred_idx])[0]
    return probs, pred_class

def predict_with_mlp(preprocessor, model_path, le, df):
    if torch is None:
        raise ImportError("Torch is required for MLP model.")
    X = preprocessor.transform(df)
    from scipy import sparse
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    state_dict = torch.load(model_path, map_location='cpu')
    first_w_key = [k for k in state_dict.keys() if k.endswith('.weight')][0]
    in_dim = state_dict[first_w_key].shape[1]
    out_dim = len(le.classes_)

    model = MLP(in_dim, out_dim)
    model.load_state_dict(state_dict)
    model.eval()

    xt = torch.from_numpy(X).to("cpu")
    with torch.no_grad():
        logits = model(xt)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = le.inverse_transform([pred_idx])[0]
    return probs, pred_class

def main():
    parser = argparse.ArgumentParser(description="Predict PE malware family for one file")
    parser.add_argument("--file", "-f", required=True, help="Path to PE file (.exe/.dll)")
    parser.add_argument("--preprocessor", default="preprocessor.joblib", help="Path to preprocessor.joblib")
    parser.add_argument("--rf", action="store_true", help="Use RandomForest model (rf_pe_model.joblib)")
    parser.add_argument("--mlp", action="store_true", help="Use PyTorch MLP model (pe_mlp.pt)")
    parser.add_argument("--model-path", default=None, help="Custom model path")
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[!] File not found: {args.file}")
        sys.exit(2)

    feats = extract_features(args.file, label="")
    if feats is None:
        print("[!] Feature extraction failed.")
        sys.exit(1)

    preprocessor = load_preprocessor(args.preprocessor)
    df = build_input_df(feats, preprocessor, debug=args.debug)

    # Load label encoder
    le = joblib.load("label_encoder.joblib")

    rf_path = args.model_path if (args.model_path and args.model_path.endswith(".joblib")) else "rf_pe_model.joblib"
    mlp_path = args.model_path if (args.model_path and args.model_path.endswith(".pt")) else "pe_mlp.pt"

    if args.rf and not args.mlp:
        chosen = "rf"
    elif args.mlp and not args.rf:
        chosen = "mlp"
    else:
        chosen = "rf" if os.path.exists(rf_path) else "mlp"

    if chosen == "rf":
        model = joblib.load(rf_path)
        probs, pred_class = predict_with_rf(preprocessor, model, le, df)
        model_name = "RandomForest"
    else:
        probs, pred_class = predict_with_mlp(preprocessor, mlp_path, le, df)
        model_name = "PyTorch MLP"

    print("=== Prediction result ===")
    print(f"File: {args.file}")
    print(f"Model used: {model_name}")
    print("Class probabilities:")
    for cls, p in zip(le.classes_, probs):
        print(f"  {cls:12s}: {p:.4f}")
    print(f"Predicted label: {pred_class}")

if __name__ == "__main__":
    main()
