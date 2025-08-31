# extract_features.py
import os
import pefile
import numpy as np
import pandas as pd


def extract_features_onehot(pe_path: str, section_list=None, dll_list=None):
    if section_list is None:
        section_list = [b".text", b".data", b".rdata", b".rsrc", b".reloc"]
    if dll_list is None:
        dll_list = ["KERNEL32.dll", "USER32.dll", "ADVAPI32.dll", "GDI32.dll"]

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

    return np.array(features, dtype=np.int8)


def process_dataset(pe_dir: str, csv_out: str):
    records = []

    for fname in os.listdir(pe_dir):
        if not fname.lower().endswith((".exe", ".dll")):
            continue

        pe_path = os.path.join(pe_dir, fname)
        feats = extract_features_onehot(pe_path)
        if feats is not None:
            records.append([fname] + feats.tolist())

    if records:
        df = pd.DataFrame(records)
        df.to_csv(csv_out, index=False, header=False)


if __name__ == "__main__":
    dataset_dir = "./dataset_pe"   # thư mục chứa file PE
    csv_file = "./features.csv"

    process_dataset(dataset_dir, csv_file)
    print("Done: extracted feature vectors to CSV")
