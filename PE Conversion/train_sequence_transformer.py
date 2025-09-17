# train_sequence_transformer.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import glob

def read_bytes_fixed(path, max_len):
    with open(path, "rb") as f:
        b = f.read(max_len)
    arr = list(b)
    if len(arr) < max_len:
        arr += [0] * (max_len - len(arr))
    return np.array(arr, dtype=np.int64)

class ByteDataset(Dataset):
    def __init__(self, file_paths, labels, max_len):
        self.file_paths = file_paths
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        arr = read_bytes_fixed(self.file_paths[idx], self.max_len)
        label = self.labels[idx]
        return torch.tensor(arr, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SeqTransformer(nn.Module):
    def __init__(self, vocab_size=256, emb_dim=64, n_layers=2, n_heads=4, dim_feedforward=256, max_len=2000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = PositionalEncoding(emb_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x: (B, L)
        emb = self.embedding(x)  # (B, L, E)
        emb = self.pos(emb)
        out = self.transformer(emb)  # (B, L, E)
        out = out.mean(dim=1)  # global avg pool
        return self.classifier(out)

def collect_file_list(dataset_dir):
    files = []
    labels = []
    for root, _, fs in os.walk(dataset_dir):
        for f in fs:
            if f.lower().endswith((".exe", ".dll")):
                p = os.path.join(root, f)
                # label by parent folder
                parent = os.path.normpath(p).split(os.sep)[-2]
                label = 0 if parent.lower() == "benign" else 1
                files.append(p)
                labels.append(label)
    return files, labels

def train(args):
    files, labels = collect_file_list(args.dataset_dir)
    if len(files) == 0:
        raise RuntimeError("No exe/dll files found in dataset_dir")

    X_train, X_val, y_train, y_val = train_test_split(files, labels, test_size=0.15, stratify=labels, random_state=42)
    train_ds = ByteDataset(X_train, y_train, args.max_len)
    val_ds = ByteDataset(X_val, y_val, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeqTransformer(vocab_size=256, emb_dim=args.emb_dim, n_layers=args.layers, n_heads=args.heads, dim_feedforward=args.ff_dim, max_len=args.max_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        # validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                preds = out.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(y.numpy().tolist())
        print("Validation report:")
        print(classification_report(y_true, y_pred, target_names=["Benign","Virus"]))

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "seq_transformer.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model -> {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset", help="Root dataset with subfolders per label")
    parser.add_argument("--out_dir", default="models_seq", help="Where to save model")
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=256)
    args = parser.parse_args()
    train(args)


#python train_sequence_transformer.py --dataset_dir dataset --out_dir models_seq --max_len 2000