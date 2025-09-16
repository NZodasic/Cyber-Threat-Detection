import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from prepare_dataset import extract_features_from_file

# Load dataset
df = pd.read_csv("malware_dataset.csv")
X = df.drop("label", axis=1).values
y = (df["label"]=="Virus").astype(int).values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model
class MalwareNet(nn.Module):
    def __init__(self):
        super(MalwareNet, self).__init__()
        self.fc1 = nn.Linear(2000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Train
model = MalwareNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):  # tăng số epoch nếu muốn
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
with torch.no_grad():
    outputs = model(X_tensor)
    preds = torch.argmax(outputs, dim=1)
print(classification_report(y_tensor, preds, target_names=["Benign", "Virus"]))

# Save model
torch.save(model.state_dict(), "malware_model.pt")
print("Model saved to malware_model.pt")

# Test on a single exe
def predict_file(model, file_path, max_len=2000):
    feat = extract_features_from_file(file_path, max_len)
    feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(feat_tensor)
        pred = torch.argmax(output, dim=1).item()
    return "Virus" if pred==1 else "Benign"

# Example usage
print("Prediction:", predict_file(model, "Dataset/Virus test/Zbot/sample1.exe"))
