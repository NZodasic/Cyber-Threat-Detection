import os
import numpy as np
from PIL import Image

def file_to_bytes(file_path: str) -> np.ndarray:
    with open(file_path, "rb") as f:
        content = f.read()
    return np.frombuffer(content, dtype=np.uint8)

def bytes_to_grayscale(byte_arr: np.ndarray) -> Image.Image:
    size = int(np.ceil(np.sqrt(len(byte_arr))))
    padded = np.pad(byte_arr, (0, size * size - len(byte_arr)), mode="constant")
    return Image.fromarray(padded.reshape((size, size)).astype(np.uint8))

def bytes_to_rgb(byte_arr: np.ndarray) -> Image.Image:
    size = int(np.ceil(np.sqrt(len(byte_arr) / 3)))
    padded = np.pad(byte_arr, (0, size * size * 3 - len(byte_arr)), mode="constant")
    return Image.fromarray(padded.reshape((size, size, 3)).astype(np.uint8))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert PE files to images")
    parser.add_argument("input", help="Path to folder containing PE files")
    parser.add_argument("output", help="Output folder for images")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for fname in os.listdir(args.input):
        fpath = os.path.join(args.input, fname)
        if not os.path.isfile(fpath):
            continue

        byte_arr = file_to_bytes(fpath)

        # Grayscale
        gray_img = bytes_to_grayscale(byte_arr)
        gray_img.save(os.path.join(args.output, f"{fname}.gray.png"))

        # RGB
        rgb_img = bytes_to_rgb(byte_arr)
        rgb_img.save(os.path.join(args.output, f"{fname}.rgb.png"))

    print(f"Saved images to {args.output}")
