import os
import pefile
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class PEFeatureExtractor:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore")

    def extract_features(self, file_path: str) -> dict:
        try:
            pe = pefile.PE(file_path, fast_load=True)
        except Exception as e:
            return {"error": str(e)}

        features = {}

        # DOS header
        features["e_magic"] = pe.DOS_HEADER.e_magic
        features["e_lfanew"] = pe.DOS_HEADER.e_lfanew

        # File header
        file_header = pe.FILE_HEADER
        features["Machine"] = file_header.Machine
        features["NumberOfSections"] = file_header.NumberOfSections
        features["TimeDateStamp"] = file_header.TimeDateStamp
        features["PointerToSymbolTable"] = file_header.PointerToSymbolTable
        features["NumberOfSymbols"] = file_header.NumberOfSymbols
        features["SizeOfOptionalHeader"] = file_header.SizeOfOptionalHeader
        features["Characteristics"] = file_header.Characteristics

        # Optional header
        opt = pe.OPTIONAL_HEADER
        features["AddressOfEntryPoint"] = opt.AddressOfEntryPoint
        features["ImageBase"] = opt.ImageBase
        features["SectionAlignment"] = opt.SectionAlignment
        features["FileAlignment"] = opt.FileAlignment
        features["SizeOfImage"] = opt.SizeOfImage
        features["Subsystem"] = opt.Subsystem

        # Section headers (chỉ lấy tên & entropy)
        for sec in pe.sections:
            features[f"Section_{sec.Name.decode(errors='ignore').strip()}_Entropy"] = sec.get_entropy()

        return features

    def encode_features(self, feature_list: list[dict]) -> np.ndarray:
        df = pd.DataFrame(feature_list).fillna(0)
        return df.values

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract PE features")
    parser.add_argument("input", help="Path to folder containing PE files")
    parser.add_argument("output", help="Output CSV file")
    args = parser.parse_args()

    extractor = PEFeatureExtractor()
    all_features = []

    for fname in os.listdir(args.input):
        fpath = os.path.join(args.input, fname)
        if not os.path.isfile(fpath):
            continue
        feats = extractor.extract_features(fpath)
        feats["filename"] = fname
        all_features.append(feats)

    df = pd.DataFrame(all_features).fillna(0)
    df.to_csv(args.output, index=False)
    print(f"Saved features to {args.output}")