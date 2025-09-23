"""
Tự động extract features + dự đoán malware/benign cho PE file

Usage:
  python predict_pe.py --file sample.exe
  python predict_pe.py --file sample.exe --mlp
  python predict_pe.py --file sample.exe --rf
  
Requirements:
  pip install pefile pandas numpy scikit-learn joblib torch
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
    """
    Extract features từ 1 file PE.
    (Hàm này copy từ pe_feature_extractor.py để file predict chạy độc lập)
    """
    try:
        pe = pefile.PE(path)
    except pefile.PEFormatError:
        print(f"[!] {path} không phải file PE hợp lệ.")
        return None

    features = {}
    # Hash MD5
    features['md5'] = hashlib.md5(open(path,'rb').read()).hexdigest()

    # Một số ví dụ features cơ bản
    features['SizeOfOptionalHeader'] = pe.OPTIONAL_HEADER.SizeOfHeaders
    features['SizeOfCode'] = pe.OPTIONAL_HEADER.SizeOfCode
    features['SizeOfImage'] = pe.OPTIONAL_HEADER.SizeOfImage
    features['NumberOfSections'] = len(pe.sections)
    features['DllCharacteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics

    # Import table
    imports = []
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll = entry.dll.decode(errors='ignore')
            for imp in entry.imports:
                if imp.name:
                    imports.append(imp.name.decode(errors='ignore'))
    imports_text = " ".join(imports)
    features['imports_text'] = imports_text

    features['label'] = label
    return features

# 2. Model + Preprocessor
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def load_preprocessor(path="preprocessor.joblib"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor file not found: {path}")
    return joblib.load(path)

def build_input_df(feats: dict, preprocessor):
    # Lấy danh sách cột từ preprocessor
    required_cols = []
    for name, trans, cols_spec in preprocessor.transformers_:
        if isinstance(cols_spec, (list, tuple)):
            required_cols.extend(cols_spec)
        else:
            required_cols.append(cols_spec)
    required_cols = list(dict.fromkeys(required_cols))

    row = {}
    for c in required_cols:
        if c in feats:
            row[c] = feats[c]
        else:
            row[c] = "" if any(s in str(c).lower() for s in ("import","dll","text")) else 0
    df = pd.DataFrame([row], columns=required_cols)
    return df

def predict_with_rf(preprocessor, model, df):
    X = preprocessor.transform(df)
    from scipy import sparse
    if sparse.issparse(X):
        X = X.toarray()
    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)
    return prob, pred

def predict_with_mlp(preprocessor, model_path, df):
    if torch is None:
        raise ImportError("Cần cài torch để dùng MLP.")
    X = preprocessor.transform(df)
    from scipy import sparse
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    state_dict = torch.load(model_path, map_location='cpu')
    first_w_key = [k for k in state_dict.keys() if k.endswith('.weight')][0]
    in_dim = state_dict[first_w_key].shape[1]
    if X.shape[1] != in_dim:
        raise ValueError(f"Mismatch feature dim. Model expects {in_dim}, input has {X.shape[1]}")
    model = MLP(in_dim)
    model.load_state_dict(state_dict)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    xt = torch.from_numpy(X).to(device)
    with torch.no_grad():
        logits = model(xt)
        probs = torch.sigmoid(logits).cpu().numpy()
    prob = float(probs.ravel()[0])
    pred = int(prob >= 0.5)
    return prob, pred

def main():
    parser = argparse.ArgumentParser(description="Predict PE malware/benign cho 1 file")
    parser.add_argument("--file", "-f", required=True, help="Path tới file PE (.exe/.dll)")
    parser.add_argument("--preprocessor", default="preprocessor.joblib", help="Đường dẫn preprocessor.joblib")
    parser.add_argument("--rf", action="store_true", help="Force dùng RandomForest model (rf_pe_model.joblib)")
    parser.add_argument("--mlp", action="store_true", help="Force dùng PyTorch MLP model (pe_mlp.pt)")
    parser.add_argument("--model-path", default=None, help="Model path tùy chọn")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[!] Không tìm thấy file: {args.file}")
        sys.exit(2)

    feats = extract_features(args.file, label="")
    if feats is None:
        print("[!] Extract features thất bại.")
        sys.exit(1)

    preprocessor = load_preprocessor(args.preprocessor)
    df = build_input_df(feats, preprocessor)

    rf_path = args.model_path if (args.model_path and args.model_path.endswith(".joblib")) else "rf_pe_model.joblib"
    mlp_path = args.model_path if (args.model_path and args.model_path.endswith(".pt")) else "pe_mlp.pt"

    if args.rf and not args.mlp:
        chosen = "rf"
    elif args.mlp and not args.rf:
        chosen = "mlp"
    else:
        if os.path.exists(rf_path):
            chosen = "rf"
        elif os.path.exists(mlp_path):
            chosen = "mlp"
        else:
            raise FileNotFoundError("Không tìm thấy model. Đặt rf_pe_model.joblib hoặc pe_mlp.pt trong thư mục làm việc.")

    if chosen == "rf":
        model = joblib.load(rf_path)
        prob, pred = predict_with_rf(preprocessor, model, df)
        model_name = "RandomForest"
    else:
        prob, pred = predict_with_mlp(preprocessor, mlp_path, df)
        model_name = "PyTorch MLP"

    label_str = "Malware" if pred == 1 else "Benign"
    print("=== Prediction result ===")
    print(f"File: {args.file}")
    print(f"Model used: {model_name}")
    print(f"Malware probability: {prob:.6f}")
    print(f"Predicted label: {label_str}")

if __name__ == "__main__":
    main()
