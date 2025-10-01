# client.py
import flwr as fl
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import MLP, get_model_parameters, set_model_parameters
from data_utils import load_preprocessed, partition_data
import random

# ------------------------
# Local training routine
# ------------------------
def train_local(model, X, y, epochs=1, batch_size=32, lr=1e-3, device="cpu"):
    model.train()
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

def evaluate_model(model, X, y, device="cpu"):
    model.eval()
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    correct, total = 0, 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.to(device)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            loss_sum += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return loss_sum/total, correct/total

# Flower client
class MalClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_data, val_data, device="cpu"):
        self.cid = cid
        self.model = model
        self.train_X, self.train_y = train_data
        self.val_X, self.val_y = val_data
        self.device = device

    def get_parameters(self):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        # set incoming global parameters
        set_model_parameters(self.model, parameters)
        # local train
        epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 32))
        lr = float(config.get("lr", 1e-3))
        train_local(self.model, self.train_X, self.train_y,
                    epochs=epochs, batch_size=batch_size, lr=lr, device=self.device)
        return get_model_parameters(self.model), len(self.train_y), {}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        loss, acc = evaluate_model(self.model, self.val_X, self.val_y, device=self.device)
        # Flower expects: loss, num_examples, {"accuracy": acc}
        return float(loss), len(self.val_y), {"accuracy": float(acc)}

# ------------------------
# CLI & main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client id (0..K-1)")
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--data_csv", type=str, default=None)
    parser.add_argument("--iid", action="store_true", help="Make IID partitioning")
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load data (global preprocessed X,y expected)
    X, y = load_preprocessed(csv_path=args.data_csv)
    clients = partition_data(X, y, n_clients=args.n_clients, iid=args.iid)
    train_X, train_y = clients[args.cid]
    # split client local train a small val for evaluate
    n = train_X.shape[0]
    if n < 10:
        val_X, val_y = train_X, train_y
    else:
        split = int(0.8*n)
        val_X, val_y = train_X[split:], train_y[split:]
        train_X, train_y = train_X[:split], train_y[:split]

    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    model = MLP(input_dim=input_dim, num_classes=num_classes)

    # Start Flower client
    client = MalClient(args.cid, model, (train_X, train_y), (val_X, val_y), device=args.device)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client,
                                 config={"local_epochs": args.local_epochs,
                                         "batch_size": 64, "lr": 1e-3})

if __name__ == "__main__":
    main()
