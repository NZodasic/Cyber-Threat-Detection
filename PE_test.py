import torch
import joblib
import pandas as pd
import pefile
import numpy as np

# ----- 1. Load model và preprocessor -----
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Load preprocessor
preprocessor = joblib.load("preprocessor.joblib")

# Khi load model phải biết input_dim (tùy thuộc vào preprocessor)
input_dim = preprocessor.transform(pd.DataFrame([{}])).shape[1]  # hack lấy số cột
model = MLP(input_dim)
model.load_state_dict(torch.load("pe_mlp.pt"))
model.eval()

# ----- 2. Extract feature từ PE file -----
def extract_pe_features(filepath):
    try:
        pe = pefile.PE(filepath)
        
        features = {
            "Machine": pe.FILE_HEADER.Machine,
            "NumberOfSections": pe.FILE_HEADER.NumberOfSections,
            "TimeDateStamp": pe.FILE_HEADER.TimeDateStamp,
            "PointerToSymbolTable": pe.FILE_HEADER.PointerToSymbolTable,
            "NumberOfSymbols": pe.FILE_HEADER.NumberOfSymbols,
            "SizeOfOptionalHeader": pe.FILE_HEADER.SizeOfOptionalHeader,
            "Characteristics": pe.FILE_HEADER.Characteristics,
            "AddressOfEntryPoint": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            "ImageBase": pe.OPTIONAL_HEADER.ImageBase,
            "SectionAlignment": pe.OPTIONAL_HEADER.SectionAlignment,
            "FileAlignment": pe.OPTIONAL_HEADER.FileAlignment,
            "SizeOfImage": pe.OPTIONAL_HEADER.SizeOfImage,
            "SizeOfHeaders": pe.OPTIONAL_HEADER.SizeOfHeaders,
            "Subsystem": pe.OPTIONAL_HEADER.Subsystem,
            "DllCharacteristics": pe.OPTIONAL_HEADER.DllCharacteristics,
            "SizeOfStackReserve": pe.OPTIONAL_HEADER.SizeOfStackReserve,
            "SizeOfHeapReserve": pe.OPTIONAL_HEADER.SizeOfHeapReserve,
        }
        
        return pd.DataFrame([features])
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

# ----- 3. Hàm predict -----
def is_malware(filepath):
    features_df = extract_pe_features(filepath)
    if features_df is None:
        return None

    # Preprocess
    X = preprocessor.transform(features_df)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred = np.argmax(probs)

    return {"prediction": "Malware" if pred == 1 else "Benign",
            "probabilities": {"Benign": probs[0], "Malware": probs[1]}}

# ----- 4. Test thử -----
result = is_malware("sample.exe")
print(result)
