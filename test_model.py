import joblib
import numpy as np

def extract_byte_features(filepath, max_len=2000):
    """
    Đọc file .exe và convert thành feature vector từ byte value.
    Pad hoặc truncate để fixed length.
    """
    with open(filepath, "rb") as f:
        bytez = f.read()
    byte_vals = [b for b in bytez]

    # Truncate or pad
    if len(byte_vals) > max_len:
        byte_vals = byte_vals[:max_len]
    else:
        byte_vals += [0] * (max_len - len(byte_vals))

    return np.array(byte_vals).reshape(1, -1)

# Load model
model = joblib.load("malware_model.pkl")

def predict_exe(filepath):
    features = extract_byte_features(filepath)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    return pred, prob

# Example usage
if __name__ == "__main__":
    file_path = "test.exe"   # file exe bất kỳ
    label, prob = predict_exe(file_path)
    print(f"Prediction: {label}, Probability: {prob}")
