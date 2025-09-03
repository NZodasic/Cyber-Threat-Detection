import os
import csv
import pefile
from collections import Counter

def analyze_dataset(input_dir, section_csv="sections_count.csv", dll_csv="dll_count.csv"):
    section_counter = Counter()
    dll_counter = Counter()
    total_files = 0

    # Duyệt toàn bộ thư mục con
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Lọc file PE bằng phần mở rộng
            if not file.lower().endswith((".exe", ".dll", ".sys", ".scr")):
                continue
            try:
                pe = pefile.PE(file_path)
                total_files += 1

                # Sections
                for section in pe.sections:
                    name = section.Name.decode(errors="ignore").strip("\x00")
                    if name:
                        section_counter[name] += 1

                # Imports
                if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                    for entry in pe.DIRECTORY_ENTRY_IMPORT:
                        dll = entry.dll.decode(errors="ignore")
                        if dll:
                            dll_counter[dll] += 1

            except Exception:
                # Bỏ qua file không đọc được
                continue

    # Ghi thống kê Sections
    with open(section_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Count", "Percentage"])
        for sec, count in section_counter.most_common():
            pct = count / total_files * 100 if total_files else 0
            writer.writerow([sec, count, f"{pct:.2f}%"])

    # Ghi thống kê DLL
    with open(dll_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["DLL", "Count", "Percentage"])
        for dll, count in dll_counter.most_common():
            pct = count / total_files * 100 if total_files else 0
            writer.writerow([dll, count, f"{pct:.2f}%"])

    print(f"Processed {total_files} PE files (from {input_dir})")
    print(f"Sections saved to {section_csv}")
    print(f"DLLs saved to {dll_csv}")


if __name__ == "__main__":
    dataset_dir = "Dataset"
    analyze_dataset(dataset_dir)
