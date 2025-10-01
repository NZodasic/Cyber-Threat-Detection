"""
Predict PE malware family (multi-class) for one file.
Standalone version: includes its own feature extractor (extended).

Usage:
  python predict.py --file sample.exe --rf
  python predict.py --file sample.exe --mlp
  python predict.py --debug
"""

import argparse
import os, sys, joblib, hashlib
import numpy as np
import pandas as pd
import pefile

# Torch (for MLP)
try:
    import torch
    import torch.nn as nn
except Exception:
    torch, nn = None, None


# ===============================
# Feature Extraction
# ===============================
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

    # --- Basic PE optional header features ---
    try:
        oh = pe.OPTIONAL_HEADER
        features['SizeOfHeaders'] = oh.SizeOfHeaders
        features['SizeOfCode'] = oh.SizeOfCode
        features['SizeOfImage'] = oh.SizeOfImage
        features['AddressOfEntryPoint'] = oh.AddressOfEntryPoint
        features['DllCharacteristics'] = oh.DllCharacteristics
        features['Subsystem'] = oh.Subsystem
        features['LinkerVersion'] = oh.MajorLinkerVersion
        features['Machine'] = pe.FILE_HEADER.Machine
        features['TimeDateStamp'] = pe.FILE_HEADER.TimeDateStamp
        features['Characteristics'] = pe.FILE_HEADER.Characteristics
        features['Checksum'] = oh.CheckSum if hasattr(oh, "CheckSum") else 0
    except Exception:
        pass

    # --- Section features ---
    try:
        sections = pe.sections
        features['NumberOfSections'] = len(sections)
        sec_entropy = [s.get_entropy() for s in sections]
        sec_rawsize = [s.SizeOfRawData for s in sections]
        sec_virtualsize = [s.Misc_VirtualSize for s in sections]

        features['mean_section_entropy'] = np.mean(sec_entropy) if sec_entropy else 0
        features['FileEntropy'] = np.mean(sec_entropy) if sec_entropy else 0
        features['mean_section_ratio'] = np.mean(
            [(r / v) if v > 0 else 0 for r, v in zip(sec_rawsize, sec_virtualsize)]
        ) if sec_rawsize else 0

        features['FileSize'] = sum(sec_rawsize) if sec_rawsize else 0

        # Section-by-section
        sec_names = [b".text", b".data", b".rdata", b".rsrc", b".bss",
                     b".edata", b".idata", b".reloc", b".tls", b".pdata", b".debug"]
        for name in sec_names:
            nm = name.decode("ascii", errors="ignore").lower().replace(".", "")
            present = any(s.Name.startswith(name) for s in sections)
            features[f"sec_present_{nm}"] = int(present)
            match_sec = [s for s in sections if s.Name.startswith(name)]
            if match_sec:
                s = match_sec[0]
                features[f"sec_entropy_{nm}"] = s.get_entropy()
                features[f"sec_rawsize_{nm}"] = s.SizeOfRawData
                features[f"sec_virtualsize_{nm}"] = s.Misc_VirtualSize
                features[f"sec_ratio_{nm}"] = (
                    (s.SizeOfRawData / s.Misc_VirtualSize) if s.Misc_VirtualSize else 0
                )
            else:
                features[f"sec_entropy_{nm}"] = 0
                features[f"sec_rawsize_{nm}"] = 0
                features[f"sec_virtualsize_{nm}"] = 0
                features[f"sec_ratio_{nm}"] = 0
    except Exception:
        pass

    # --- Imports (DLL + API count) ---
    imports = []
    dlls = []
    try:
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                try:
                    dll = entry.dll.decode(errors="ignore").lower()
                except Exception:
                    dll = str(entry.dll)
                dlls.append(dll)
                for imp in entry.imports:
                    if imp.name:
                        try:
                            imports.append(imp.name.decode(errors="ignore"))
                        except Exception:
                            imports.append(str(imp.name))
    except Exception:
        pass

    features["imports_text"] = " ".join(imports)
    features["import_count"] = len(imports)
    features["dll_kernel32_dll"] = int("kernel32.dll" in dlls)
    features["dll_user32_dll"] = int("user32.dll" in dlls)
    features["dll_advapi32_dll"] = int("advapi32.dll" in dlls)
    features["dll_gdi32_dll"] = int("gdi32.dll" in dlls)
    features["dll_ntdll_dll"] = int("ntdll.dll" in dlls)
    features["dll_wininet_dll"] = int("wininet.dll" in dlls)
    features["dll_ws2_32_dll"] = int("ws2_32.dll" in dlls)
    features["dll_wsock32_dll"] = int("wsock32.dll" in dlls)

    # --- Resources ---
    try:
        if hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
            features["resource_count"] = len(pe.DIRECTORY_ENTRY_RESOURCE.entries)
            total_size = sum(
                [entry.data.struct.Size for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries if hasattr(entry, "data")]
            )
            features["resource_total_size"] = total_size
        else:
            features["resource_count"] = 0
            features["resource_total_size"] = 0
    except Exception:
        features["resource_count"] = 0
        features["resource_total_size"] = 0

    # --- Dummy suspicious counters ---
    features["suspicious_api_count"] = 0
    features["suspicious_string_count"] = 0

    features["label"] = label
    return features


# ===============================
# Torch MLP Model
# ===============================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)


# ===============================
# Prediction helpers
# ===============================
def build_input_df(feats, preprocessor):
    # match columns with training preprocessor
    required_cols = []
    for name, trans, cols in preprocessor.transformers_:
        if isinstance(cols, (list, tuple)):
            required_cols.extend(cols)
        else:
            required_cols.append(cols)

    row = {}
    for c in required_cols:
        if c in feats:
            row[c] = feats[c]
        else:
            row[c] = "" if "import" in c or "dll" in c else 0
    df = pd.DataFrame([row], columns=required_cols)
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

    state_dict = torch.load(model_path, map_location="cpu")
    first_w_key = [k for k in state_dict.keys() if k.endswith(".weight")][0]
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


# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Predict PE malware family")
    parser.add_argument("--file", "-f", required=True, help="PE file path")
    parser.add_argument("--preprocessor", default="preprocessor.joblib", help="Path to preprocessor")
    parser.add_argument("--rf", action="store_true", help="Use RandomForest")
    parser.add_argument("--mlp", action="store_true", help="Use PyTorch MLP")
    parser.add_argument("--model-path", default=None, help="Custom model path")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[!] File not found: {args.file}")
        sys.exit(2)

    feats = extract_features(args.file)
    if feats is None:
        sys.exit(1)

    preprocessor = joblib.load(args.preprocessor)
    le = joblib.load("label_encoder.joblib")
    df = build_input_df(feats, preprocessor)

    rf_path = args.model_path if (args.model_path and args.model_path.endswith(".joblib")) else "rf_pe_model.joblib"
    mlp_path = args.model_path if (args.model_path and args.model_path.endswith(".pt")) else "pe_mlp.pt"

    if args.rf:
        model = joblib.load(rf_path)
        probs, pred_class = predict_with_rf(preprocessor, model, le, df)
        model_name = "RandomForest"
    else:
        probs, pred_class = predict_with_mlp(preprocessor, mlp_path, le, df)
        model_name = "PyTorch MLP"

    print("=== Prediction result ===")
    print(f"File: {args.file}")
    print(f"Model used: {model_name}")
    for cls, p in zip(le.classes_, probs):
        print(f"  {cls:12s}: {p:.4f}")
    print(f"Predicted label: {pred_class}")


if __name__ == "__main__":
    main()
