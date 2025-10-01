# data_utils.py
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import json
import os

def load_preprocessed(csv_path=None, preprocessor_path=None, feature_order_path=None):
    """
    If you have a CSV already (features extracted), use it. Otherwise,
    expect preprocessor.joblib + feature_order.json available.
    This function will:
      - load CSV if csv_path provided
      - otherwise try to load X.npy/y.npy if present
    Returns X (np.ndarray), y (np.ndarray), label_encoder (if available or None)
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Expect 'label' column for target. Adjust if different.
        assert 'label' in df.columns, "CSV must have 'label' column"
        y = df['label'].values
        X = df.drop(columns=['label']).values
        return X.astype(np.float32), y
    # fallback: look for X.npy and y.npy
    if os.path.exists("X.npy") and os.path.exists("y.npy"):
        X = np.load("X.npy").astype(np.float32)
        y = np.load("y.npy")
        return X, y
    raise FileNotFoundError("Provide csv_path or prepare X.npy & y.npy")

def partition_data(X, y, n_clients=3, iid=True, seed=42):
    """
    Return a list of (X_i, y_i) for each client.
    If iid=True: random split roughly equal.
    If iid=False: do a simple label-wise partitioning to simulate non-iid.
    """
    np.random.seed(seed)
    N = X.shape[0]
    idxs = np.arange(N)
    if iid:
        np.random.shuffle(idxs)
        parts = np.array_split(idxs, n_clients)
    else:
        # Non-IID: sort by label, split contiguous blocks (clients get concentrated labels)
        order = np.argsort(y)
        parts = np.array_split(order, n_clients)
    clients = []
    for p in parts:
        clients.append((X[p], y[p]))
    return clients

def make_global_testset(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
