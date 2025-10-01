import sys
import struct

def check_file_type(path):
    with open(path, "rb") as f:
        data = f.read(0x1000)  # read first 4KB (headers)

    # 1. Must start with "MZ"
    if not data.startswith(b"MZ"):
        return "Not a PE file (missing MZ header)"

    # 2. e_lfanew offset at 0x3C → points to NT headers
    if len(data) < 0x40:
        return "Too small to be a valid PE"

    e_lfanew = struct.unpack("<I", data[0x3C:0x40])[0]
    if e_lfanew >= len(data):
        return f"Invalid e_lfanew ({e_lfanew}), not a PE"

    # 3. Must contain "PE\0\0" signature
    if data[e_lfanew:e_lfanew+4] != b"PE\0\0":
        # could be NE (16-bit Windows) or LE (old Linear Executable)
        sig = data[e_lfanew:e_lfanew+4]
        if sig.startswith(b"NE"):
            return "NE file (16-bit New Executable, not supported by pefile)"
        elif sig.startswith(b"LE") or sig.startswith(b"LX"):
            return "LE/LX file (old OS/2 format)"
        else:
            return f"Unknown signature {sig} at {hex(e_lfanew)}"

    # 4. Machine type (PE32 vs PE32+)
    # IMAGE_FILE_HEADER is 20 bytes, Optional header follows
    machine = struct.unpack("<H", data[e_lfanew+4:e_lfanew+6])[0]
    magic = struct.unpack("<H", data[e_lfanew+24:e_lfanew+26])[0]

    if magic == 0x10B:
        return "PE32 (32-bit executable)"
    elif magic == 0x20B:
        return "PE32+ (64-bit executable)"
    else:
        return f"PE file with unknown optional header magic {hex(magic)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_pe_type.py <file.exe>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        result = check_file_type(path)
        print(f"{path}: {result}")
    except Exception as e:
        print(f"Error reading {path}: {e}")
