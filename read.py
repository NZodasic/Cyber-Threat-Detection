import pefile

# Mở file PE
pe = pefile.PE("sample.exe")

# Duyệt qua tất cả section
for section in pe.sections:
    print("Name:", section.Name.decode(errors="ignore").strip("\x00"))
    print("Virtual Address:", hex(section.VirtualAddress))
    print("Virtual Size:", section.Misc_VirtualSize)
    print("Raw Size:", section.SizeOfRawData)
    print("MD5 Hash:", section.get_hash_md5())
    print("-" * 40)
