# train_image_cnn.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import glob
import numpy as np

class ImageFolderDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8,8)),
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.net(x)

def gather_image_paths(data_dir):
    classes = []
    files = []
    labels = []
    # expecting structure: data_dir/<label>/*.png
    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        class_idx = 0 if label_name.lower() == "benign" else 1
        for f in glob.glob(os.path.join(label_dir, "*.png")):
            files.append(f)
            labels.append(class_idx)
            classes.append(label_name)
    return files, labels

def train(args):
    files, labels = gather_image_paths(args.data_dir)
    if len(files) == 0:
        raise RuntimeError("No images found. Make sure images are in data_dir/<label>/*.png")

    X_train, X_val, y_train, y_val = train_test_split(files, labels, test_size=0.15, stratify=labels, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    train_ds = ImageFolderDataset(X_train, y_train, transform=transform)
    val_ds = ImageFolderDataset(X_val, y_val, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(in_ch=1, n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, labs in train_loader:
            imgs = imgs.to(device)
            labs = labs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] train loss: {total_loss/len(train_loader):.4f}")

        # validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                preds = out.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(labs.numpy().tolist())
        print("Validation report:")
        print(classification_report(y_true, y_pred, target_names=["Benign","Virus"]))

    # save
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "image_cnn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model -> {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="converted/images", help="Root image dir (subfolders per label)")
    parser.add_argument("--out_dir", default="models", help="Where to save model")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=128)
    args = parser.parse_args()
    train(args)
