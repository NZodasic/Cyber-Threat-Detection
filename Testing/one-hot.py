# extract_features.py
import os
import pefile
import numpy as np
import pandas as pd


def extract_features_onehot(pe_path: str, section_list=None, dll_list=None):
    if section_list is None:
        section_list = [b".text", b".rdata", b".data", b".idata", b".edata", b".pdata", b".rsrc", b".reloc", b".bss", b".tls", b".debug"]
    if dll_list is None:
        dll_list = ["KERNEL32.dll", "ADVAPI32.dll", "USER32.dll", "GDI32.dll", "NTDLL.dll", "WSOCK32.dll", "WS2_32.dll", "WININET.dll"]

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
    section_list = [b".text", b".rdata", b".data", b".idata", b".edata", b".pdata", b".rsrc", b".reloc", b".bss", b".tls", b".debug"]
    dll_list = ["KERNEL32.dll", "ADVAPI32.dll", "USER32.dll", "GDI32.dll", "NTDLL.dll", "WSOCK32.dll", "WS2_32.dll", "WININET.dll"]

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
    csv_file = "Report/Feature/features.csv"

    process_dataset(dataset_dir, csv_file)
    print("Done: extracted feature vectors to CSV")
