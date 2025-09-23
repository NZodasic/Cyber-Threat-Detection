# train_dl.py
import os
import argparse
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# -------------------------
# Config / Utilities
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -------------------------
# Data handling
# -------------------------
def load_and_prepare(csv_path, label_col="Label", drop_cols=None, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load CSV, drop columns that are non-numeric (or in drop_cols), label-encode label,
    return X_train, X_val, X_test (numpy), y_train, y_val, y_test, scaler, label_encoder
    """
    df = pd.read_csv(csv_path)
    original_cols = list(df.columns)

    # drop specified columns (like MD5)
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])

    # Ensure label exists
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV")

    # Separate label
    y = df[label_col].astype(str)  # ensure string for LabelEncoder
    X = df.drop(columns=[label_col])

    # Convert LinkerVersion if exists and is string like "14.0" -> float
    if "LinkerVersion" in X.columns:
        try:
            X["LinkerVersion"] = X["LinkerVersion"].astype(float)
        except:
            # try parsing
            X["LinkerVersion"] = X["LinkerVersion"].apply(lambda v: float(str(v)) if pd.notna(v) else 0.0)

    # Keep only numeric columns (drop strings like import_0.. which may be present)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric features found after cleaning. Need numeric features for this pipeline.")
    X_numeric = X[numeric_cols].fillna(0.0)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # First split train+val and test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_numeric.values, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    # Split train and val
    val_frac_of_tmp = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac_of_tmp, random_state=random_state, stratify=y_tmp
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    meta = {
        "feature_names": numeric_cols,
        "original_columns": original_cols
    }

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            scaler, le, meta)

# -------------------------
# PyTorch dataset
# -------------------------
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# Model (simple MLP)
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256,128], dropout=0.3, num_classes=2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------
# Training loop
# -------------------------
def train_loop(model, opt, criterion, train_loader, val_loader, epochs, device, model_dir, early_stop_patience=10):
    writer = SummaryWriter(log_dir=os.path.join(model_dir, "runs"))
    best_val_loss = float("inf")
    best_epoch = -1
    trigger = 0

    for epoch in range(1, epochs+1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * Xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += Xb.size(0)

        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * Xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += Xb.size(0)
        val_loss /= (total if total else 1)
        val_acc = correct / total if total else 0.0

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s")

        # Early stopping / save best
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            trigger = 0
            # save model
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
            print(f"[+] Saved best_model.pt (epoch {epoch})")
        else:
            trigger += 1
            if trigger >= early_stop_patience:
                print("[!] Early stopping triggered.")
                break

    writer.close()
    print(f"[+] Training finished. Best epoch: {best_epoch}, best_val_loss: {best_val_loss:.4f}")

# -------------------------
# Evaluate on test
# -------------------------
def evaluate_model(model, model_dir, X_test, y_test, device, batch_size=256):
    model.eval()
    ds = TabularDataset(X_test, y_test)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    preds_all = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            preds_all.append(preds)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
    acc = correct / total if total else 0.0
    preds_all = np.concatenate(preds_all) if preds_all else np.array([])
    print(f"[+] Test Accuracy: {acc:.4f} ({correct}/{total})")
    return acc, preds_all

# -------------------------
# Main train function
# -------------------------
def run_training(csv_path, out_dir, epochs=100, batch_size=256, lr=1e-3, hidden=[256,128], dropout=0.3):
    os.makedirs(out_dir, exist_ok=True)

    print("[*] Loading and preparing data...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, label_encoder, meta) = load_and_prepare(csv_path,
                                                     drop_cols=["MD5"] if "MD5" in pd.read_csv(csv_path).columns else None)

    # Save preprocessors & meta
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(out_dir, "label_encoder.joblib"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Build datasets
    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    test_ds = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    print(f"Input dim: {input_dim}, num_classes: {num_classes}")

    model = MLP(input_dim=input_dim, hidden_dims=hidden, dropout=dropout, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, verbose=True)

    # Train
    train_loop(model, opt, criterion, train_loader, val_loader, epochs, DEVICE, out_dir, early_stop_patience=12)

    # Load best model
    best_path = os.path.join(out_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    else:
        print("[!] best_model.pt not found, using current model state.")

    # Evaluate test
    test_acc, preds = evaluate_model(model, out_dir, X_test, y_test, DEVICE, batch_size=batch_size)

    # Save final model (script-friendly)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden": hidden,
        "dropout": dropout,
        "num_classes": num_classes
    }, os.path.join(out_dir, "model_package.pt"))
    print(f"[+] Saved model package to {os.path.join(out_dir, 'model_package.pt')}")
    return out_dir

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to features CSV")
    parser.add_argument("--out", type=str, required=True, help="Output folder prefix for model (e.g. model_basic)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", nargs="+", type=int, default=[256,128])
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    run_training(args.csv, args.out, epochs=args.epochs, batch_size=args.batch, lr=args.lr, hidden=args.hidden, dropout=args.dropout)

# python train_dl.py --csv pe_features_dataset.csv --out model_basic
# python train_dl.py --csv pe_features_extended.csv --out model_extended
