import os, collections
root = "Dataset"
cnt = collections.Counter()
for dirpath, _, files in os.walk(root):
    for f in files:
        if f.lower().endswith(".exe"):
            # parent label: giả sử folder 1 cấp trên là label (Benign hoặc Virus)
            label = os.path.basename(os.path.dirname(dirpath)) if os.path.basename(dirpath) else os.path.basename(dirpath)
            cnt[label] += 1
print(cnt)
