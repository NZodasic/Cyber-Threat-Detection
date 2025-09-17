# pe_to_images.py
import os
import math
import numpy as np
from PIL import Image


def pe_to_grayscale_image(pe_path: str, out_path: str, width: int = 256):
    with open(pe_path, "rb") as f:
        data = f.read()

    arr = np.frombuffer(data, dtype=np.uint8)
    height = math.ceil(len(arr) / width)
    arr = np.pad(arr, (0, height * width - len(arr)), 'constant', constant_values=0)
    arr = arr.reshape((height, width))

    img = Image.fromarray(arr, mode='L')
    img.save(out_path)


def pe_to_rgb_image(pe_path: str, out_path: str, width: int = 256):
    with open(pe_path, "rb") as f:
        data = f.read()

    arr = np.frombuffer(data, dtype=np.uint8)
    pad_len = (3 - len(arr) % 3) % 3
    arr = np.pad(arr, (0, pad_len), 'constant', constant_values=0)

    arr = arr.reshape((-1, 3))
    height = math.ceil(len(arr) / width)
    arr = np.pad(arr, ((0, height * width - len(arr)), (0, 0)), 'constant', constant_values=0)
    arr = arr.reshape((height, width, 3))

    img = Image.fromarray(arr, mode='RGB')
    img.save(out_path)


def process_dataset(pe_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(pe_dir):
        if not fname.lower().endswith((".exe", ".dll")):
            continue

        pe_path = os.path.join(pe_dir, fname)
        base = os.path.splitext(fname)[0]

        gray_out = os.path.join(out_dir, base + "_gray.png")
        rgb_out = os.path.join(out_dir, base + "_rgb.png")

        try:
            pe_to_grayscale_image(pe_path, gray_out)
            pe_to_rgb_image(pe_path, rgb_out)
        except Exception as e:
            print(f"[!] Error converting {fname} to image: {e}")


if __name__ == "__main__":
    dataset_dir = "./dataset_pe"   # thư mục chứa file PE
    output_dir = "./output_images"

    process_dataset(dataset_dir, output_dir)
    print("Saved grayscale & RGB images")
