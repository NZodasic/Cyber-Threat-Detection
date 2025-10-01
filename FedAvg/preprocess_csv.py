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

    # Load raw data
    df = pd.read_csv(args.input)
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column.")

    # Separate X, y
    y_raw = df["label"].values
    X_raw = df.drop(columns=["label"])

    import_cols = [c for c in X_raw.columns if c.startswith("import_")]
    if import_cols and "imports_text" not in X_raw.columns:
        X_raw["imports_text"] = X_raw[import_cols].fillna("").agg(" ".join, axis=1)

    # Load preprocessor + label encoder
    preprocessor = joblib.load(args.preprocessor)
    label_encoder = joblib.load(args.label_encoder)

    # Transform features
    X_proc = preprocessor.transform(X_raw)

    # Encode labels
    y_enc = label_encoder.transform(y_raw)

    # Convert to DataFrame for saving
    X_df = pd.DataFrame(X_proc, columns=[f"f{i}" for i in range(X_proc.shape[1])])
    X_df["label"] = y_enc

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    X_df.to_csv(args.output, index=False)
    print(f"[OK] Saved processed dataset to {args.output}")
    print(f"Shape: {X_proc.shape}, Classes: {len(label_encoder.classes_)}")

if __name__ == "__main__":
    main()

# python preprocess_csv.py --input raw_features.csv \
#     --preprocessor preprocessor.joblib \
#     --label_encoder label_encoder.joblib \
#     --output data/features.csv
