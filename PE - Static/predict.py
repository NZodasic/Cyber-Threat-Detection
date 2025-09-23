"""
Extract features + dự đoán malware/benign cho PE file (độc lập)

Usage:
  python predict.py --file sample.exe
  python predict.py --file sample.exe --mlp
  python predict.py --file sample.exe --rf
  python predict.py --file sample.exe --debug   # show debug info about columns

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
    """
    Extract features từ 1 file PE (ví dụ cơ bản).
    Nếu bạn có thêm features trong pe_feature_extractor gốc thì copy vào đây.
    """
    try:
        pe = pefile.PE(path)
    except pefile.PEFormatError:
        print(f"[!] {path} không phải file PE hợp lệ.")
        return None

    features = {}
    # Hash MD5
    with open(path, "rb") as f:
        b = f.read()
    features['md5'] = hashlib.md5(b).hexdigest()

    # Ví dụ features cơ bản (bạn có thể thêm/bớt)
    try:
        features['SizeOfOptionalHeader'] = pe.OPTIONAL_HEADER.SizeOfHeaders
        features['SizeOfCode'] = pe.OPTIONAL_HEADER.SizeOfCode
        features['SizeOfImage'] = pe.OPTIONAL_HEADER.SizeOfImage
        features['NumberOfSections'] = len(pe.sections)
        features['DllCharacteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
    except Exception:
        # nếu bất kỳ trường nào không có, gán 0
        features.setdefault('SizeOfOptionalHeader', 0)
        features.setdefault('SizeOfCode', 0)
        features.setdefault('SizeOfImage', 0)
        features.setdefault('NumberOfSections', 0)
        features.setdefault('DllCharacteristics', 0)

    # Import table -> text
    imports = []
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            try:
                dll = entry.dll.decode(errors='ignore')
            except Exception:
                dll = str(entry.dll)
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

# helper: unwrap pipeline to list of estimators
def _flatten_transformer(trans):
    """
    Return list of estimator objects by unwrapping Pipeline recursively.
    """
    try:
        from sklearn.pipeline import Pipeline
    except Exception:
        Pipeline = None

    if Pipeline is not None and isinstance(trans, Pipeline):
        out = []
        for _name, step in trans.steps:
            out.extend(_flatten_transformer(step))
        return out
    else:
        return [trans]

def _is_numeric_estimator(est):
    """
    Heuristics: estimator is numeric if:
      - has attribute 'strategy' (SimpleImputer) OR
      - class name contains known numeric indicators
    """
    name = est.__class__.__name__.lower()
    if hasattr(est, 'strategy'):
        return True
    numeric_indicators = (
        'simpleimputer', 'imputer', 'scaler', 'standardscaler', 'minmaxscaler',
        'robustscaler', 'powertransformer', 'quantiletransformer', 'pca',
        'polynomialfeatures'
    )
    for k in numeric_indicators:
        if k in name:
            return True
    return False

def _is_text_estimator(est):
    """
    Heuristics: estimator is text-ish if:
      - class name has 'vector' / 'tfidf' / 'count' OR
      - has attribute 'vocabulary_' (fitted CountVectorizer/Tfidf)
    """
    name = est.__class__.__name__.lower()
    if 'vector' in name or 'tfidf' in name or 'count' in name or 'hashing' in name:
        return True
    if hasattr(est, 'vocabulary_'):
        return True
    return False

def build_input_df(feats: dict, preprocessor, debug=False):
    """
    Build DataFrame correct kiểu cho preprocessor:
      - detect numeric cols vs text cols by inspecting preprocessor.transformers_
      - numeric missing -> 0, text missing -> ""
      - coerce numeric cols to numeric to avoid SimpleImputer median error on ''
    """
    num_cols = []
    text_cols = []
    unknown_cols = []

    for name, trans, cols_spec in preprocessor.transformers_:
        # ignore 'remainder' definitions that may appear
        cols = cols_spec if isinstance(cols_spec, (list, tuple)) else [cols_spec]
        # some pipelines: unwrap
        estimators = []
        try:
            estimators = _flatten_transformer(trans)
        except Exception:
            estimators = [trans]

        # decide type by heuristics across estimators
        if any(_is_numeric_estimator(est) for est in estimators):
            num_cols.extend(cols)
        elif any(_is_text_estimator(est) for est in estimators):
            text_cols.extend(cols)
        else:
            # fallback heuristic by column name
            for c in cols:
                cstr = str(c).lower()
                if any(k in cstr for k in ('import', 'dll', 'text', 'name', 'api')):
                    text_cols.append(c)
                elif any(k in name.lower() for k in ('num', 'numeric', 'cont')):
                    num_cols.append(c)
                else:
                    unknown_cols.append(c)

    # for unknown columns: assume numeric unless name obviously text
    for c in unknown_cols:
        cstr = str(c).lower()
        if any(k in cstr for k in ('import', 'dll', 'text', 'name', 'api')):
            text_cols.append(c)
        else:
            num_cols.append(c)

    # dedupe while preserving order
    def _unique(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    num_cols = _unique(num_cols)
    text_cols = _unique(text_cols)
    required_cols = num_cols + text_cols

    # build row with safe defaults
    row = {}
    for c in required_cols:
        if c in feats:
            row[c] = feats[c]
        else:
            row[c] = "" if c in text_cols else 0

    df = pd.DataFrame([row], columns=required_cols)

    # === important: coerce numeric cols to numeric (no empty strings) ===
    for c in num_cols:
        if c in df.columns:
            # convert to numeric; if conversion fails -> 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # ensure text columns are strings (CountVectorizer expects str)
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna('')

    if debug:
        print("DEBUG: classified columns:")
        print("  numeric cols:", num_cols)
        print("  text cols   :", text_cols)
        print("  final df dtypes:")
        print(df.dtypes)
        print("  final df preview:")
        print(df.head(1).to_dict(orient='records')[0])

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
    parser.add_argument("--debug", action="store_true", help="Show debug info about columns / dtypes")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[!] Không tìm thấy file: {args.file}")
        sys.exit(2)

    feats = extract_features(args.file, label="")
    if feats is None:
        print("[!] Extract features thất bại.")
        sys.exit(1)

    preprocessor = load_preprocessor(args.preprocessor)
    df = build_input_df(feats, preprocessor, debug=args.debug)

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
