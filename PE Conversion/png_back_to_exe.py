# png_back_to_exe.py
import os
import json
import argparse
import numpy as np
from PIL import Image

def png_to_bytes(png_path, meta_path):
    """Read PNG and meta JSON and return original bytes (truncated to original_len)."""
    # Load meta
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "original_len" not in meta:
        raise ValueError(f"meta JSON does not contain 'original_len': {meta_path}")

    orig_len = int(meta["original_len"])

    # Load image and flatten
    img = Image.open(png_path).convert("L")
    arr = np.array(img, dtype=np.uint8).flatten()

    # Truncate to original length
    arr = arr[:orig_len]
    return bytes(arr), meta

def reconstruct(png_path, meta_path, out_dir):
    b, meta = png_to_bytes(png_path, meta_path)
    os.makedirs(out_dir, exist_ok=True)
    # Derive output filename from meta original_file if present else from png filename
    out_name = None
    if "original_file" in meta:
        out_name = os.path.basename(meta["original_file"])
    else:
        base = os.path.splitext(os.path.basename(png_path))[0]
        out_name = base + ".restored.exe"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "wb") as f:
        f.write(b)
    print(f"[OK] Reconstructed -> {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Reconstruct exe from PNG + meta JSON")
    parser.add_argument("--png", required=True, help="Path to PNG image")
    parser.add_argument("--meta", required=True, help="Path to meta JSON")
    parser.add_argument("--out", default="reconstructed", help="Output folder")
    args = parser.parse_args()
    reconstruct(args.png, args.meta, args.out)

if __name__ == "__main__":
    main()

#python png_back_to_exe.py --png path/to/img.png --meta path/to/meta.json --out out_folder/
