import torch
import torch.nn as nn
import joblib
import pandas as pd
import sys

from extract_pe_features_extended import extract_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define MLP same as training ---
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

def predict_pe(file_path: str):
    # 1. Extract features
    feats = extract_features(file_path, label="Unknown")
    if feats is None:
        print("[-] Feature extraction failed.")
        return

    # Drop label (if present)
    if "label" in feats:
        feats.pop("label")

    # Convert to DataFrame for preprocessor
    df = pd.DataFrame([feats])

    # 2. Load preprocessor
    preprocessor = joblib.load("preprocessor.joblib")
    X = preprocessor.transform(df)

    # 3. Convert to torch tensor
    if hasattr(X, "toarray"):  # sparse
        X = X.toarray()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    # 4. Load model
    model = MLP(X_t.shape[1]).to(device)
    model.load_state_dict(torch.load("pe_mlp.pt", map_location=device))
    model.eval()

    # 5. Prediction
    with torch.no_grad():
        logits = model(X_t)
        prob = torch.sigmoid(logits).item()
        pred = "Malware" if prob >= 0.5 else "Benign"

    print(f"[+] File: {file_path}")
    print(f"    Probability(Malware) = {prob:.4f}")
    print(f"    Prediction = {pred}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pe_file.py <path_to_pe_file>")
        sys.exit(1)

    predict_pe(sys.argv[1])
