import os
import numpy as np
import pandas as pd

def extract_features_from_file(file_path, max_len=2000):
    with open(file_path, "rb") as f:
        byte_arr = list(f.read(max_len))
    # pad nếu file ngắn hơn
    if len(byte_arr) < max_len:
        byte_arr += [0] * (max_len - len(byte_arr))
    return byte_arr[:max_len]

def build_dataset(root_dir, max_len=2000):
    data = []
    labels = []

    for label in ["Benign", "Virus"]:
        folder = os.path.join(root_dir, label)
        for subdir, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".exe"):
                    file_path = os.path.join(subdir, file)
                    features = extract_features_from_file(file_path, max_len)
                    data.append(features)
                    labels.append(label)
    return np.array(data), np.array(labels)

if __name__ == "__main__":
    X, y = build_dataset("Dataset", max_len=2000)
    df = pd.DataFrame(X)
    df["label"] = y
    df.to_csv("malware_dataset.csv", index=False)
    print("Dataset saved to malware_dataset.csv")
