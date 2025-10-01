# server.py
import flwr as fl
import argparse
import numpy as np
import torch
from model import MLP, get_model_parameters, set_model_parameters
from data_utils import load_preprocessed, make_global_testset
from sklearn.metrics import accuracy_score, f1_score
import time

def evaluate_global(weights, X_test, y_test, device="cpu"):
    input_dim = X_test.shape[1]
    num_classes = len(np.unique(y_test))
    model = MLP(input_dim=input_dim, num_classes=num_classes)
    # set weights
    set_model_parameters(model, weights)
    model.eval()
    import torch.nn.functional as F
    with torch.no_grad():
        X = torch.from_numpy(X_test).float().to(device)
        logits = model(X)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num_clients", type=int, default=3)
    args = parser.parse_args()

    X, y = load_preprocessed(csv_path=args.data_csv)
    X_train, X_test, y_train, y_test = make_global_testset(X, y, test_size=0.2)
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    # initialize model
    model = MLP(input_dim=input_dim, num_classes=num_classes)

    # Define Flower strategy: FedAvg (default), with evaluation function
    def evaluate_fn(server_round, parameters, config):
        # parameters is a List[np.ndarray]
        acc, f1 = evaluate_global(parameters, X_test, y_test)
        print(f"[Server] Round {server_round} global eval — acc: {acc:.4f}  macro-F1: {f1:.4f}")
        return float(1.0 - acc), {"accuracy": float(acc), "macro_f1": float(f1)}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,      # fraction of clients used for training each round
        fraction_eval=1.0,
        min_fit_clients=args.num_clients,
        min_eval_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_fn=evaluate_fn,
    )

    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=args.rounds),
                           server_address="0.0.0.0:8080",
                           strategy=strategy)

if __name__ == "__main__":
    main()
