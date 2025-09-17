# merge_features.py
import os
import math
import json
import argparse
import pefile
import numpy as np
import pandas as pd
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64
from collections import Counter
import hashlib

# Config vocabularies (tweak as needed)
SECTION_LIST = [b".text", b".rdata", b".data", b".idata", b".edata", b".pdata", b".rsrc", b".reloc", b".bss", b".tls", b".debug"]
DLL_LIST = ["KERNEL32.dll", "ADVAPI32.dll", "USER32.dll", "GDI32.dll", "NTDLL.dll", "WS2_32.dll", "WSOCK32.dll", "WININET.dll"]
# Choose common mnemonics; you can extend this list
OPCODE_LIST = ["mov", "push", "pop", "call", "jmp", "cmp", "test", "add", "sub", "xor", "and", "or", "lea", "nop", "ret"]

def safe_pe_load(path):
    try:
        return pefile.PE(path)
    except Exception:
        return None

def get_header_features(pe):
    """Extract numeric header fields of interest"""
    opt = pe.OPTIONAL_HEADER
    fh = pe.FILE_HEADER
    return {
        "Machine": fh.Machine,
        "NumberOfSections": fh.NumberOfSections,
        "TimeDateStamp": fh.TimeDateStamp,
        "AddressOfEntryPoint": opt.AddressOfEntryPoint,
        "SizeOfImage": opt.SizeOfImage,
        "Subsystem": opt.Subsystem,
        "Characteristics": fh.Characteristics
    }

def get_section_features(pe):
    names = [s.Name.rstrip(b'\x00') for s in pe.sections]
    onehot = {sec.decode(errors="ignore"): (1 if sec in names else 0) for sec in SECTION_LIST}
    sizes = {}
    entropies = {}
    for s in pe.sections:
        name = s.Name.rstrip(b'\x00').decode(errors="ignore")
        sizes[f"{name}_VirtualSize"] = s.Misc_VirtualSize
        sizes[f"{name}_SizeOfRawData"] = s.SizeOfRawData
        # entropy
        try:
            data = s.get_data()
            probs = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256) / max(1, len(data))
            probs = probs[probs>0]
            entropy = -np.sum(probs * np.log2(probs))
        except Exception:
            entropy = 0.0
        entropies[f"{name}_entropy"] = float(entropy)
    return onehot, sizes, entropies

def get_imports_features(pe):
    dlls = []
    funcs = []
    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            try:
                dlls.append(entry.dll.decode(errors="ignore"))
                for imp in entry.imports:
                    if imp.name:
                        funcs.append(imp.name.decode(errors="ignore"))
            except Exception:
                continue
    dll_onehot = {d: (1 if d in dlls else 0) for d in DLL_LIST}
    # top-k imported function counts could be added; here only presence
    return dll_onehot, {"num_imports": len(funcs)}

def extract_opcodes_hist(pe, opcode_vocab=OPCODE_LIST, max_bytes=200_000):
    """Disassemble .text (if present) and count mnemonics"""
    counts = Counter()
    arch = CS_MODE_32
    try:
        # detect 64-bit
        if pe.PE_TYPE == pefile.OPTIONAL_HEADER_MAGIC_PE_PLUS:
            arch = CS_MODE_64
    except Exception:
        arch = CS_MODE_32
    try:
        text_sec = None
        for s in pe.sections:
            if s.Name.startswith(b".text"):
                text_sec = s
                break
        if text_sec is None:
            return {op: 0 for op in opcode_vocab}
        code = text_sec.get_data()[:max_bytes]
        base = pe.OPTIONAL_HEADER.ImageBase + text_sec.VirtualAddress
        md = Cs(CS_ARCH_X86, arch)
        for i in md.disasm(code, base):
            counts[i.mnemonic.lower()] += 1
    except Exception:
        pass
    total = sum(counts.values()) or 1
    # return normalized freq for opcode_vocab
    return {op: counts[op] / total for op in opcode_vocab}

def entropy_of_bytes(data_bytes):
    if not data_bytes:
        return 0.0
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    probs = np.bincount(arr, minlength=256) / arr.size
    probs = probs[probs>0]
    return float(-np.sum(probs * np.log2(probs)))

def process_file(path):
    pe = safe_pe_load(path)
    if pe is None:
        return None
    rec = {}
    rec["Filename"] = os.path.basename(path)
    # header fields
    hdr = get_header_features(pe)
    rec.update(hdr)
    # sections
    onehot_sec, sizes_sec, ent_sec = get_section_features(pe)
    rec.update(onehot_sec)
    rec.update(sizes_sec)
    rec.update(ent_sec)
    # imports
    dll_onehot, impinfo = get_imports_features(pe)
    rec.update(dll_onehot)
    rec.update(impinfo)
    # opcode hist
    op_hist = extract_opcodes_hist(pe)
    rec.update(op_hist)
    # file-level entropy and size
    try:
        with open(path, "rb") as f:
            b = f.read()
            rec["file_entropy"] = entropy_of_bytes(b)
            rec["file_size"] = len(b)
            rec["sha256"] = hashlib.sha256(b).hexdigest()
    except Exception:
        rec["file_entropy"] = 0.0
        rec["file_size"] = 0
        rec["sha256"] = ""
    return rec

def build_dataset(input_dir, out_csv):
    records = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith((".exe", ".dll")):
                continue
            fpath = os.path.join(root, fname)
            try:
                rec = process_file(fpath)
                if rec:
                    # derive label from parent folder (assumes dataset/benign, dataset/malware)
                    parent = os.path.normpath(fpath).split(os.sep)[-2]
                    rec["label"] = parent
                    records.append(rec)
                    print(f"[OK] {fname}")
            except Exception as e:
                print(f"[ERR] {fname}: {e}")
    if records:
        df = pd.DataFrame(records)
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(records)} records to {out_csv}")
    else:
        print("No records found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge features from PE files (header/sections/imports/opcodes)")
    parser.add_argument("--input_dir", default="dataset", help="Input dataset folder that contains subfolders for classes")
    parser.add_argument("--out_csv", default="merged_features.csv", help="Output CSV path")
    args = parser.parse_args()
    build_dataset(args.input_dir, args.out_csv)

#python merge_features.py --input_dir dataset --out merged_features.csv
