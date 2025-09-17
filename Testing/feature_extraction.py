import os
import math
import json
import numpy as np
from PIL import Image


def pe_to_grayscale_image(pe_path: str, out_path: str, meta_out: str, width: int = 256):
    with open(pe_path, "rb") as f:
        data = f.read()

    arr = np.frombuffer(data, dtype=np.uint8)
    height = math.ceil(len(arr) / width)
    pad_len = height * width - len(arr)
    arr = np.pad(arr, (0, pad_len), 'constant', constant_values=0)
    arr = arr.reshape((height, width))

    img = Image.fromarray(arr, mode='L')
    img.save(out_path)

    meta = {"original_length": len(data), "width": width, "pad_len": pad_len}
    with open(meta_out, "w") as f:
        json.dump(meta, f)


def pe_to_rgb_image(pe_path: str, out_path: str, meta_out: str, width: int = 256):
    with open(pe_path, "rb") as f:
        data = f.read()

    arr = np.frombuffer(data, dtype=np.uint8)
    pad_len = (3 - len(arr) % 3) % 3
    arr = np.pad(arr, (0, pad_len), 'constant', constant_values=0)
    arr = arr.reshape((-1, 3))

    height = math.ceil(len(arr) / width)
    pad_rows = height * width - len(arr)
    arr = np.pad(arr, ((0, pad_rows), (0, 0)), 'constant', constant_values=0)
    arr = arr.reshape((height, width, 3))

    img = Image.fromarray(arr, mode='RGB')
    img.save(out_path)

    meta = {"original_length": len(data), "width": width, "pad_len": pad_len + pad_rows * 3}
    with open(meta_out, "w") as f:
        json.dump(meta, f)


def reconstruct_from_image(img_path: str, meta_path: str, out_pe: str):
    img = Image.open(img_path)
    arr = np.array(img)
    arr = arr.flatten()

    with open(meta_path, "r") as f:
        meta = json.load(f)

    byte_arr = arr[: meta["original_length"]]
    with open(out_pe, "wb") as f:
        f.write(byte_arr.tobytes())


def process_dataset(pe_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(pe_dir):
        if not fname.lower().endswith((".exe", ".dll")):
            continue

        pe_path = os.path.join(pe_dir, fname)
        base = os.path.splitext(fname)[0]

        gray_out = os.path.join(out_dir, base + "_gray.png")
        gray_meta = os.path.join(out_dir, base + "_gray.json")
        rgb_out = os.path.join(out_dir, base + "_rgb.png")
        rgb_meta = os.path.join(out_dir, base + "_rgb.json")

        try:
            pe_to_grayscale_image(pe_path, gray_out, gray_meta)
            pe_to_rgb_image(pe_path, rgb_out, rgb_meta)
        except Exception as e:
            print(f"[!] Error converting {fname}: {e}")


if __name__ == "__main__":
    dataset_dir = "./dataset_pe"
    output_dir = "./output_images"
    process_dataset(dataset_dir, output_dir)
    print("Saved grayscale & RGB images + metadata")
