# model.py
import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        return self.fc3(x)

# Helpers to convert between PyTorch state_dict and list of numpy arrays (Flower expects List[np.ndarray])
def get_model_parameters(model: torch.nn.Module):
    """Return model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model: torch.nn.Module, parameters):
    """Set model parameters from a list of NumPy ndarrays."""
    params_dict = dict(zip(model.state_dict().keys(), parameters))
    # Convert numpy -> tensor
    state_dict = {k: torch.tensor(v) for k, v in params_dict.items()}
    model.load_state_dict(state_dict)
