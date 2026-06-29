import json
from collections import Counter
from pathlib import Path

data = json.loads(Path("data/physcibench.json").read_text(encoding="utf-8"))
print("count", len(data))
print("keys", list(data[0].keys()))
print("types", Counter(r["type"] for r in data))
print("categories", Counter(r["category"] for r in data))
all_files = []
for r in data:
    all_files.extend(r.get("files") or [])
print("unique files", len(set(all_files)))
exts = Counter(Path(f).suffix.lower() for f in set(all_files))
print("extensions", dict(exts))
print("sample record:")
r = data[0]
for k, v in r.items():
    if isinstance(v, str) and len(v) > 120:
        print(f"  {k}: {v[:120]}...")
    else:
        print(f"  {k}: {v}")
