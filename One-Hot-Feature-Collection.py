import os
import pefile
import numpy as np
import pandas as pd


def load_section_and_dll_lists(section_csv: str, dll_csv: str):
    df_sections = pd.read_csv(section_csv)
    section_list = [s.encode() for s in df_sections["Section"].tolist() if isinstance(s, str)]

    df_dlls = pd.read_csv(dll_csv)
    dll_list = [dll for dll in df_dlls["DLL"].tolist() if isinstance(dll, str)]

    return section_list, dll_list


def extract_features_onehot(pe_path: str, section_list, dll_list):
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


def process_dataset(pe_dir: str, section_csv: str, dll_csv: str, csv_out: str):
    records = []
    section_list, dll_list = load_section_and_dll_lists(section_csv, dll_csv)

    for fname in os.listdir(pe_dir):
        if not fname.lower().endswith((".exe", ".dll")):
            continue

        pe_path = os.path.join(pe_dir, fname)
        feats = extract_features_onehot(pe_path, section_list, dll_list)
        if feats is not None:
            records.append([fname] + feats.tolist())

    if records:
        header = ["Filename"] + [sec.decode() for sec in section_list] + dll_list
        df = pd.DataFrame(records, columns=header)
        df.to_csv(csv_out, index=False)


if __name__ == "__main__":
    dataset_dir = "Dataset_Minimized/Benign"   # thư mục chứa file PE
    section_csv = "sections_count.csv"         # file thống kê sections từ analyze_dataset.py
    dll_csv = "dll_count.csv"                  # file thống kê DLL từ analyze_dataset.py
    csv_file = "Report/Feature/features.csv"

    process_dataset(dataset_dir, section_csv, dll_csv, csv_file)
    print("Done: extracted feature vectors to CSV")
