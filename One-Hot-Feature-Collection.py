import os
import pefile
import numpy as np
import pandas as pd
import math
import hashlib


def section_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = freq / len(data)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def extract_features_onehot(pe_path: str, section_list=None, dll_list=None):
    if section_list is None:
        section_list = [b".text", b".rdata", b".data", b".idata", b".rsrc", b".reloc"]
    if dll_list is None:
        dll_list = ["KERNEL32.dll", "ADVAPI32.dll", "USER32.dll", "GDI32.dll"]

    try:
        pe = pefile.PE(pe_path)
    except Exception:
        return None

    features = []

    # One-hot sections
    section_names = [s.Name.strip(b'\x00') for s in pe.sections]
    for sec in section_list:
        features.append(1 if sec in section_names else 0)

    # One-hot DLL imports
    dlls = []
    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        dlls = [entry.dll.decode(errors="ignore") for entry in pe.DIRECTORY_ENTRY_IMPORT]
    for dll in dll_list:
        features.append(1 if dll in dlls else 0)

    # Header numeric features
    try:
        features.append(pe.FILE_HEADER.NumberOfSections)
        features.append(pe.OPTIONAL_HEADER.SizeOfImage)
        features.append(pe.FILE_HEADER.TimeDateStamp)
    except Exception:
        features.extend([0, 0, 0])

    # Entropy per section
    for sec in pe.sections:
        features.append(round(section_entropy(sec.get_data()), 4))

    return np.array(features, dtype=np.float32)


def process_dataset(pe_dir: str, csv_out: str, npy_out: str):
    records = []
    npy_data = []
    section_list = [b".text", b".rdata", b".data", b".idata", b".rsrc", b".reloc"]
    dll_list = ["KERNEL32.dll", "ADVAPI32.dll", "USER32.dll", "GDI32.dll"]

    for fname in os.listdir(pe_dir):
        if not fname.lower().endswith((".exe", ".dll")):
            continue
        pe_path = os.path.join(pe_dir, fname)
        feats = extract_features_onehot(pe_path, section_list, dll_list)
        if feats is not None:
            records.append([fname] + feats.tolist())
            npy_data.append(feats)

    if records:
        header = ["Filename"] + [sec.decode() for sec in section_list] + dll_list \
                 + ["NumSections", "SizeOfImage", "TimeDateStamp"]
        # Thêm entropy headers động
        entropy_headers = [f"Entropy_sec{i}" for i in range(len(records[0]) - len(header) - 1)]
        df = pd.DataFrame(records, columns=header + entropy_headers)
        df.to_csv(csv_out, index=False)

        np.save(npy_out, np.stack(npy_data))


if __name__ == "__main__":
    dataset_dir = "Dataset_Minimized/Benign"
    csv_file = "Report/Feature/features.csv"
    npy_file = "Report/Feature/features.npy"

    process_dataset(dataset_dir, csv_file, npy_file)
    print("Done: extracted feature vectors to CSV and NPY")
