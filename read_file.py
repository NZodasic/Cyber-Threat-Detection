import os
import csv

def detect_file_type(file_path):
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
        if header.startswith(b"MZ"):
            return "PE"
        elif header.startswith(b"PK"):
            return "ZIP"
        elif header.startswith(b"%PDF"):
            return "PDF"
        elif header.startswith(b"\x7fELF"):
            return "ELF"
        else:
            return "Unknown"
    except Exception as e:
        return f"Error: {e}"

def scan_dataset(root_dir, output_csv="file_types.csv"):
    results = []
    for root, _, files in os.walk(root_dir):
        for name in files:
            path = os.path.join(root, name)
            file_type = detect_file_type(path)
            results.append((path, file_type))

    # Ghi kết quả ra CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File", "Type"])
        writer.writerows(results)

    print(f"Scan xong. Kết quả lưu tại {output_csv}")

if __name__ == "__main__":
    dataset_dir = "/home/raymond/Desktop/MalwareAnalysis/Dataset_test/Data"
    scan_dataset(dataset_dir)
