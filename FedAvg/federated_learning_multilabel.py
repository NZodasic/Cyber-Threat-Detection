
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import matplotlib.pyplot as plt
import copy
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load PE malware dataset and prepare for multi-label classification
    
    Args:
        filepath: Path to CSV file
        test_size: Proportion for test set (default 0.2 = 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, class_names, scaler, mlb)
    """
    print("="*70)
    print("LOADING AND PREPARING DATA")
    print("="*70)
    
    # Load data
    df = pd.read_csv(filepath, index_col=0)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Separate features and labels
    # Assuming 'label' column contains malware family names
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain 'label' column")
    
    labels = df['label'].values
    
    # Drop non-feature columns
    drop_cols = ['MD5', 'label'] + [col for col in df.columns if col.startswith('import_')]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    print(f"Features shape: {X.shape}")
    
    # Convert labels to multi-label format
    # For multi-label: split comma-separated labels or create binary matrix
    # Here we'll create multi-hot encoding from single labels (can be extended)
    mlb = MultiLabelBinarizer()
    
    # Convert single labels to list format for MLB
    label_lists = [[label] for label in labels]
    y_multi = mlb.fit_transform(label_lists)
    
    print(f"Number of classes: {len(mlb.classes_)}")
    print(f"Classes: {mlb.classes_}")
    print(f"Labels shape: {y_multi.shape}")
    
    # Train-test split (fixed 80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multi, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Data preparation completed!")
    print("="*70)
    
    return X_train, X_test, y_train, y_test, feature_cols, mlb.classes_, scaler, mlb


# =============================================================================
# 2. CLIENT DATA DISTRIBUTION FUNCTIONS
# =============================================================================

def split_data_iid(X: np.ndarray, y: np.ndarray, num_clients: int, 
                   random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into IID (Independent and Identically Distributed) clients
    
    Args:
        X: Feature matrix
        y: Label matrix
        num_clients: Number of clients
        random_state: Random seed
        
    Returns:
        List of (X_client, y_client) tuples
    """
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Split indices into roughly equal parts
    client_indices = np.array_split(indices, num_clients)
    client_data = []
    
    for idx in client_indices:
        client_data.append((X[idx], y[idx]))
    
    return client_data


def split_data_non_iid(X: np.ndarray, y: np.ndarray, num_clients: int, 
                       random_state: int = 42, alpha: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into Non-IID clients using Dirichlet distribution
    
    Args:
        X: Feature matrix
        y: Label matrix  
        num_clients: Number of clients
        random_state: Random seed
        alpha: Dirichlet concentration parameter (lower = more imbalanced)
        
    Returns:
        List of (X_client, y_client) tuples
    """
    np.random.seed(random_state)
    
    # Use Dirichlet distribution để tạo tỷ lệ phân phối
    proportions = np.random.dirichlet([alpha] * num_clients)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    client_data = []
    start_idx = 0
    
    for i, prop in enumerate(proportions):
        if i == num_clients - 1:  # Client cuối cùng nhận hết phần còn lại
            end_idx = len(X)
        else:
            end_idx = start_idx + int(prop * len(X))
        
        client_indices = indices[start_idx:end_idx]
        client_data.append((X[client_indices], y[client_indices]))
        start_idx = end_idx
    
    return client_data


def split_data_random_imbalanced(X: np.ndarray, y: np.ndarray, num_clients: int,
                                random_state: int = 42, min_samples: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split data randomly with imbalanced distribution
    
    Args:
        X: Feature matrix
        y: Label matrix
        num_clients: Number of clients  
        random_state: Random seed
        min_samples: Minimum samples per client
        
    Returns:
        List of (X_client, y_client) tuples
    """
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Tạo random split points
    split_points = sorted(np.random.choice(
        range(min_samples, len(X) - min_samples * num_clients),
        num_clients - 1,
        replace=False
    ))
    split_points = [0] + split_points + [len(X)]
    
    client_data = []
    for i in range(num_clients):
        client_indices = indices[split_points[i]:split_points[i+1]]
        client_data.append((X[client_indices], y[client_indices]))
    
    return client_data


# =============================================================================
# 3. CNN MODEL FOR MULTI-LABEL CLASSIFICATION  
# =============================================================================

class MultiLabelMalwareCNN(nn.Module):
    """
    1D CNN for Multi-Label Malware Classification
    
    Architecture:
        - Conv1D layers for feature extraction
        - Fully connected layers for classification
        - Sigmoid activation for multi-label output
    """
    
    def __init__(self, feature_dim: int, num_classes: int):
        super(MultiLabelMalwareCNN, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # 1D Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Calculate flattened size
        self.flat_size = self._get_flat_size()
        
        # Fully connected layers - output là num_classes cho multi-label
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)  # Multi-label output
        )
    
    def _get_flat_size(self):
        """Calculate size after conv layers"""
        x = torch.randn(1, 1, self.feature_dim)
        x = self.conv_layers(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, features)
            
        Returns:
            logits: Raw output logits for BCEWithLogitsLoss
        """
        # Reshape for conv1d: (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)  # Raw logits cho BCEWithLogitsLoss
        return logits


# =============================================================================
# 4. FEDERATED LEARNING TRAINING FUNCTIONS
# =============================================================================

def train_client(model, train_loader, criterion, optimizer, device, epochs=1):
    """
    Train model on single client
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU/GPU)
        epochs: Number of local epochs (default 1 = 1 round = 1 epoch)
        
    Returns:
        Tuple of (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy cho multi-label (threshold=0.5)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == batch_y).all(dim=1).sum().item()
            total += batch_y.size(0)
    
    avg_loss = total_loss / (len(train_loader) * epochs)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate model on test set
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for testing
        criterion: Loss function
        device: Device (CPU/GPU)
        
    Returns:
        Tuple of (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == batch_y).all(dim=1).sum().item()
            total += batch_y.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def fedavg_aggregate(global_model, client_models, client_weights=None):
    """
    FedAvg: Aggregate client models using weighted averaging
    
    Args:
        global_model: Global model to update
        client_models: List of client models
        client_weights: List of weights (e.g., number of samples per client)
        
    Returns:
        Updated global model
    """
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    
    # Normalize weights
    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]
    
    global_dict = global_model.state_dict()
    
    # Initialize with zeros
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key], dtype = torch.float32)
    
    # Weighted sum of client parameters
    for client_model, weight in zip(client_models, client_weights):
        for key in global_dict.keys():
            global_dict[key] += client_model.state_dict()[key].float() * weight
    
    global_model.load_state_dict(global_dict)
    return global_model


# =============================================================================
# 5. MAIN FEDERATED TRAINING LOOP
# =============================================================================

def federated_training(
    client_data: List[Tuple[np.ndarray, np.ndarray]],
    test_data: Tuple[np.ndarray, np.ndarray],
    feature_dim: int,
    num_classes: int,
    num_rounds: int = 10,
    local_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = 'cpu'
):
    """
    Main Federated Learning training loop
    
    Args:
        client_data: List of (X_client, y_client) tuples
        test_data: Tuple of (X_test, y_test)
        feature_dim: Number of input features
        num_classes: Number of output classes
        num_rounds: Number of communication rounds
        local_epochs: Number of local epochs per round (default 1)
        batch_size: Batch size for training
        lr: Learning rate
        device: Device (CPU/GPU)
        
    Returns:
        Tuple of (global_model, training_history)
    """
    num_clients = len(client_data)
    
    # Initialize global model
    global_model = MultiLabelMalwareCNN(feature_dim, num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # Prepare test data
    X_test, y_test = test_data
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Tracking metrics
    history = {
        'client_losses': [[] for _ in range(num_clients)],
        'client_accuracies': [[] for _ in range(num_clients)],
        'global_test_losses': [],
        'global_test_accuracies': [],
        'client_sizes': [len(X) for X, y in client_data]
    }
    
    print("="*70)
    print("STARTING FEDERATED LEARNING TRAINING")
    print("="*70)
    print(f"Number of clients: {num_clients}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Local epochs per round: {local_epochs}")
    print(f"Client data sizes: {history['client_sizes']}")
    print("="*70)
    
    # Federated training rounds
    for round_idx in range(num_rounds):
        print(f"--- Round {round_idx + 1}/{num_rounds} ---")
        
        client_models = []
        client_weights = []
        
        # Train each client
        for client_idx, (X_client, y_client) in enumerate(client_data):
            # Create client model (copy from global)
            client_model = copy.deepcopy(global_model)
            optimizer = optim.Adam(client_model.parameters(), lr=lr)
            
            # Prepare client data
            client_dataset = TensorDataset(
                torch.tensor(X_client, dtype=torch.float32),
                torch.tensor(y_client, dtype=torch.float32)
            )
            client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
            
            # Train client
            loss, acc = train_client(
                client_model, client_loader, criterion, optimizer, device, local_epochs
            )
            
            # Log metrics
            history['client_losses'][client_idx].append(loss)
            history['client_accuracies'][client_idx].append(acc)
            
            print(f"  Client {client_idx + 1}: Loss={loss:.4f}, Acc={acc:.4f}, Samples={len(X_client)}")
            
            client_models.append(client_model)
            client_weights.append(len(X_client))
        
        # Aggregate models (FedAvg)
        global_model = fedavg_aggregate(global_model, client_models, client_weights)
        
        # Evaluate global model on test set
        test_loss, test_acc = evaluate_model(global_model, test_loader, criterion, device)
        history['global_test_losses'].append(test_loss)
        history['global_test_accuracies'].append(test_acc)
        
        print(f"  Global Test: Loss={test_loss:.4f}, Acc={test_acc:.4f}")
    
    print("="*70)
    print("FEDERATED LEARNING TRAINING COMPLETED!")
    print("="*70)
    
    return global_model, history


# =============================================================================
# 6. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_fl_metrics(history, num_clients, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Client Losses
    ax1 = axes[0, 0]
    for i in range(num_clients):
        ax1.plot(history['client_losses'][i], label=f'Client {i+1}', marker='o')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.set_title('Client Training Loss per Round')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Client Accuracies
    ax2 = axes[0, 1]
    for i in range(num_clients):
        ax2.plot(history['client_accuracies'][i], label=f'Client {i+1}', marker='o')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Client Training Accuracy per Round')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Global Test Loss
    ax3 = axes[1, 0]
    ax3.plot(history['global_test_losses'], label='Test Loss', marker='s', 
             color='red', linewidth=2)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Loss')
    ax3.set_title('Global Model Test Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Global Test Accuracy
    ax4 = axes[1, 1]
    ax4.plot(history['global_test_accuracies'], label='Test Accuracy', marker='s',
             color='green', linewidth=2)
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Global Model Test Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


# =============================================================================
# 7. MAIN EXECUTION EXAMPLE
# =============================================================================

def main():
    """
    Main execution function
    """
    # Configuration
    DATA_PATH = "/home/raymond/Dataset/DataAnalyzed/pe_features_multi.csv"
    NUM_CLIENTS = 5
    NUM_ROUNDS = 10
    LOCAL_EPOCHS = 1  # 1 round = 1 epoch
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # 1. Load and prepare data
    X_train, X_test, y_train, y_test, feature_names, class_names, scaler, mlb = load_and_prepare_data(
        DATA_PATH, test_size=0.2
    )
    
    # 2. Split data among clients (Non-IID example)
    client_data = split_data_non_iid(X_train, y_train, NUM_CLIENTS, alpha=0.5)
    
    print("Client data distribution:")
    for i, (X_c, y_c) in enumerate(client_data):
        print(f"  Client {i+1}: {len(X_c)} samples")
    
    # 3. Run Federated Learning
    global_model, history = federated_training(
        client_data=client_data,
        test_data=(X_test, y_test),
        feature_dim=X_train.shape[1],
        num_classes=len(class_names),
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        device=DEVICE
    )
    
    # 4. Plot results
    plot_fl_metrics(history, NUM_CLIENTS, save_path='fl_training_results.png')
    plt.show()
    
    print("Training completed! Check 'fl_training_results.png' for plots.")
    
    return global_model, history


if __name__ == "__main__":
    model, history = main()
