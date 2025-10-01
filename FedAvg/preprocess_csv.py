#!/usr/bin/env python3
import argparse
import pandas as pd
import joblib
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to raw extracted features CSV (from extract_pe_features_extended.py)")
    parser.add_argument("--preprocessor", type=str, default="preprocessor.joblib",
                        help="Path to saved preprocessor.joblib")
    parser.add_argument("--label_encoder", type=str, default="label_encoder.joblib",
                        help="Path to saved label_encoder.joblib")
    parser.add_argument("--output", type=str, default="data/features.csv",
                        help="Output path for processed features CSV")
    args = parser.parse_args()

    # Load raw features
    df = pd.read_csv(args.input)
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column.")

    # Separate X, y
    y_raw = df["label"].values
    X_raw = df.drop(columns=["label"])

    # Nếu có các cột import_0..import_9 thì tạo imports_text
    import_cols = [c for c in X_raw.columns if c.startswith("import_")]
    if import_cols and "imports_text" not in X_raw.columns:
        X_raw["imports_text"] = X_raw[import_cols].fillna("").agg(" ".join, axis=1)

    # Load preprocessor và label encoder
    preprocessor = joblib.load(args.preprocessor)
    label_encoder = joblib.load(args.label_encoder)

    # Transform features
    try:
        X_proc = preprocessor.transform(X_raw)
    except Exception as e:
        print("[ERROR] Preprocessor transform failed. Columns available:", X_raw.columns.tolist())
        raise e

    # Encode labels
    y_enc = label_encoder.transform(y_raw)

    # Xuất ra DataFrame
    feature_names = [f"f{i}" for i in range(X_proc.shape[1])]
    X_df = pd.DataFrame(X_proc, columns=feature_names)
    X_df["label"] = y_enc

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    X_df.to_csv(args.output, index=False)
    print(f"[OK] Saved processed dataset to {args.output}")
    print(f"Shape: {X_proc.shape}, Classes: {len(label_encoder.classes_)}")

if __name__ == "__main__":
    main()
