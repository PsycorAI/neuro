"""data_probe.py -- 2026-10-02. Inspect the corpus format before building the v4 mix: special-token usage
(doc separators), and which libraries/tokenizer are available."""
import numpy as np, importlib, glob, os
d = np.memmap("/home/glenn/projects/bdh/data/tokenized/train.bin", dtype=np.uint32, mode="r")
print("tokens", len(d))
for off in (0, len(d) // 2):
    s = np.asarray(d[off:off + 20_000_000])
    for t in (128000, 128001, 128009):
        print(f"  @{off}: id {t} count {(s == t).sum()}  (per 1M: {(s == t).sum() / 20:.0f})")
s = np.asarray(d[:4096])
print("first 40 ids:", s[:40].tolist())
idx = np.where(np.asarray(d[:2_000_000]) == 128001)[0][:3]
for i in idx:
    print("around EOS", i, np.asarray(d[i - 3:i + 4]).tolist())
for m in ("datasets", "transformers", "tokenizers", "pyarrow", "huggingface_hub"):
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", e)
for p in glob.glob("/home/glenn/**/tokenizer.json", recursive=True)[:10]:
    print("tokenizer.json:", p)
