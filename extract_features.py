# extract_features.py
import os
import csv
import pefile
import argparse

def extract_features_from_pe(file_path: str) -> dict:
    """
    Extract basic features from a PE file.
    """
    try:
        pe = pefile.PE(file_path)
    except Exception:
        return None

    features = {}
    try:
        features["entry_point"] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        features["image_base"] = pe.OPTIONAL_HEADER.ImageBase
        features["num_sections"] = len(pe.sections)
        features["size_of_code"] = pe.OPTIONAL_HEADER.SizeOfCode
        features["size_of_headers"] = pe.OPTIONAL_HEADER.SizeOfHeaders
        features["characteristics"] = pe.FILE_HEADER.Characteristics
    except Exception:
        pass

    # Section names one-hot
    for section in pe.sections:
        name = section.Name.decode(errors="ignore").strip("\x00")
        features[f"section_{name}"] = 1

    # Imports (DLL + APIs)
    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll_name = entry.dll.decode(errors="ignore") if entry.dll else "UNKNOWN"
            features[f"dll_{dll_name}"] = 1
            for imp in entry.imports:
                try:
                    if imp.name:
                        api_name = imp.name.decode(errors="ignore")
                        features[f"api_{api_name}"] = 1
                    else:
                        features[f"api_ordinal_{imp.ordinal}"] = 1
                except Exception:
                    continue

    return features

def walk_dataset(dataset_dir: str):
    """
    Walk dataset folder and yield (file_path, label).
    """
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith((".exe", ".dll")):
                label = "Benign" if "Benign" in root else "Virus"
                yield os.path.join(root, f), label

def main(dataset_dir: str, output_csv: str):
    rows = []
    for file_path, label in walk_dataset(dataset_dir):
        feats = extract_features_from_pe(file_path)
        if feats:
            feats["filename"] = os.path.basename(file_path)
            feats["label"] = label
            rows.append(feats)

    # Collect all feature keys
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(list(fieldnames))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PE features to CSV")
    parser.add_argument("--dataset", required=True, help="Path to dataset root folder")
    parser.add_argument("--output", default="Processed/features.csv", help="Output CSV file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    main(args.dataset, args.output)
