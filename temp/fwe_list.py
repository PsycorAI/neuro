"""List FineWeb-Edu parquet shards (metadata only, no download) to size the v4 mix top-up."""
from huggingface_hub import HfApi
api = HfApi()
info = api.dataset_info("HuggingFaceFW/fineweb-edu", files_metadata=True)
by = {}
for s in info.siblings:
    if s.rfilename.endswith(".parquet"):
        top = "/".join(s.rfilename.split("/")[:2])
        n, b = by.get(top, (0, 0)); by[top] = (n + 1, b + (s.size or 0))
for k in sorted(by):
    if k.startswith("sample") or len(by) < 40:
        print(f"{k:40s} files {by[k][0]:5d}  {by[k][1]/1e9:8.1f} GB")
ex = [s for s in info.siblings if s.rfilename.startswith("sample/100BT/")][:3]
for s in ex:
    print(s.rfilename, s.size)
