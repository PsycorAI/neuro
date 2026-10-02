"""build_corpus_v4.py -- 2026-10-02 (Glenn OK 10-02): v4 pretraining corpus = old corpus + a FineWeb-Edu top-up.

Why (docs/v4_plan_review_2026_10_01.md F4): the old 14.53B-token mix is 38% DCLM / 22% FineWeb-Edu and used only
3.3B of the 9.68B FineWeb-Edu tokens available. At a fixed token budget, FineWeb-Edu is the strongest lever on
knowledge/reasoning benchmarks. Adding ~8B new FineWeb-Edu tokens (HuggingFaceFW/fineweb-edu sample/100BT,
de-duplicated against the old corpus) gives ~22.5B tokens at ~50% FineWeb-Edu / 25% DCLM / 25% the rest, so a
16B-token v4 run sees each token at most once.

Layout of the new train.bin (uint32 Llama-3 ids, docs start with BOS 128000, as the old file):
    old[:split]  +  FineWeb-Edu top-up  +  old[split:]
where split = int(len(old) * 0.999) is the old val boundary. Training samples windows uniformly at random, so no
shuffle is needed. Use with  data_bin: <out>/train.bin  and  data_val_tokens: <len(old) - split>  so the val
windows are exactly the old ones (val_ppl stays comparable with every earlier run).

Phases (each resumable; progress in <out>/progress.json):
  hashes    hash the first 24 tokens of every old document (BOS-delimited) -> old_doc_hashes.npy
  fwe       per shard: download parquet -> tokenize (Llama-3 tokenizer.json, local cache) -> drop docs whose hash
            is in the old set -> append [BOS] + ids to fwe_topup.bin -> delete the parquet; stop at --target tokens
  assemble  write train.bin = old[:split] + fwe_topup.bin + old[split:], then meta.json
Usage: python scripts/build_corpus_v4.py [--target 8.0e9] [--phase all|hashes|fwe|assemble]
WATCH: tail -f /home/glenn/projects/neuro/logs/build_corpus_v4.log
"""
import argparse, json, os, time
import numpy as np

OLD = "/home/glenn/projects/bdh/data/tokenized/train.bin"
OUT = "/home/glenn/projects/bdh/data/tokenized_v4"
REPO, SUB = "HuggingFaceFW/fineweb-edu", "sample/100BT"
BOS, L = 128000, 24
P = np.uint64(0x9E3779B97F4A7C15)


def load_tok():
    """Llama-3 BPE (128,256 ids; Llama-3/3.1/3.2 share it) straight from the cached tokenizer.json."""
    import glob
    from tokenizers import Tokenizer
    cands = [p for m in ("models--meta-llama--Llama-3.2-3B-Instruct", "models--NousResearch--Meta-Llama-3-8B",
                         "models--meta-llama--Llama-3.1-8B", "models--meta-llama--Llama-3.2-1B")
             for p in sorted(glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{m}/snapshots/*/tokenizer.json")))
             if os.path.exists(p)]                     # skips dangling cache symlinks
    return Tokenizer.from_file(cands[0])


def encode(tok, texts):
    return [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def hash_rows(W):
    """W (n, L) uint64 token windows (zero-padded) -> (n,) uint64 polynomial hashes (wrapping arithmetic)."""
    h = np.zeros(W.shape[0], dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(W.shape[1]):
            h = h * P + W[:, j] + np.uint64(1)
    return h


def load_progress():
    p = f"{OUT}/progress.json"
    return json.load(open(p)) if os.path.exists(p) else {"done_shards": [], "bytes": 0, "stats": {}}


def save_progress(pr):
    tmp = f"{OUT}/progress.json.tmp"
    json.dump(pr, open(tmp, "w"), indent=1)
    os.replace(tmp, f"{OUT}/progress.json")


def phase_hashes():
    out = f"{OUT}/old_doc_hashes.npy"
    if os.path.exists(out):
        log(f"hashes: exists ({out})"); return
    d = np.memmap(OLD, dtype=np.uint32, mode="r")
    n, step, hs = len(d), 256 * 1024 * 1024, []
    for lo in range(0, n, step):
        chunk = np.asarray(d[lo:min(n, lo + step + L + 1)])
        pos = np.flatnonzero(chunk[:min(step, n - lo)] == BOS)
        idx = pos[:, None] + 1 + np.arange(L)[None, :]
        W = np.where(idx < len(chunk), chunk[np.minimum(idx, len(chunk) - 1)], 0).astype(np.uint64)
        W[np.cumsum(W == BOS, axis=1) > 0] = 0          # stop at the next document
        hs.append(hash_rows(W))
        if (lo // step) % 8 == 0:
            log(f"hashes: {lo / n:.0%} ({sum(len(h) for h in hs):,} docs)")
    h = np.unique(np.concatenate(hs))
    np.save(out, h)
    log(f"hashes: {len(h):,} unique old-document hashes -> {out}")


WIN_CURL = "/mnt/c/Windows/System32/curl.exe"
WIN_DL = ("/mnt/g/tmp/fwe_v4", "G:\\tmp\\fwe_v4")          # same folder seen from WSL / from Windows


def fetch(fn, size):
    """Download one shard (`size` = published byte size). WSL's NAT networking measured ~0.3 MB/s vs ~1.2 MB/s on
    the Windows side (10-02), so use Windows curl.exe into G:\\tmp when available (resumable). Completion is judged
    by the file reaching `size`: the WSL interop wrapper did not return after curl.exe exited (10-02 02:24 hang)."""
    from huggingface_hub import hf_hub_download
    import subprocess
    if os.path.exists(WIN_CURL):
        base = os.path.basename(fn)
        dst = f"{WIN_DL[0]}/{base}"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{fn}"
        os.makedirs(WIN_DL[0], exist_ok=True)
        for attempt in range(8):
            if os.path.exists(dst) and os.path.getsize(dst) == size:
                return dst
            p = subprocess.Popen([WIN_CURL, "-sSL", "--retry", "5", "-C", "-", "-o", f"{WIN_DL[1]}\\{base}", url],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            last, t_last = -1, time.time()
            while True:
                time.sleep(10)
                cur = os.path.getsize(dst) if os.path.exists(dst) else 0
                if cur == size:
                    p.kill(); return dst
                if cur != last:
                    last, t_last = cur, time.time()
                elif time.time() - t_last > 600:          # no growth for 10 min: curl died or hung
                    p.kill(); break
            log(f"fetch {fn}: stalled at {last / 1e9:.2f}/{size / 1e9:.2f} GB, retry {attempt + 1}")
            time.sleep(30)
        raise IOError(f"download failed: {fn}")
    return hf_hub_download(REPO, fn, repo_type="dataset", local_dir=f"{OUT}/dl")


def phase_fwe(target):
    from huggingface_hub import HfApi
    from concurrent.futures import ThreadPoolExecutor
    import pyarrow.parquet as pq
    tok = load_tok()
    old_h = np.load(f"{OUT}/old_doc_hashes.npy")
    sizes = {s.rfilename: s.size for s in HfApi().dataset_info(REPO, files_metadata=True).siblings
             if s.rfilename.startswith(SUB + "/") and s.rfilename.endswith(".parquet")}
    files = sorted(sizes)
    pr = load_progress()
    path = f"{OUT}/fwe_topup.bin"
    with open(path, "ab") as f:                          # drop any partial shard written before a crash
        f.truncate(pr["bytes"])
    todo = [fn for fn in files if fn not in pr["done_shards"]]
    pool = ThreadPoolExecutor(1)                         # prefetch the next shard while tokenizing this one
    fut = pool.submit(fetch, todo[0], sizes[todo[0]]) if todo else None
    for j, fn in enumerate(todo):
        if pr["bytes"] // 4 >= target:
            break
        t0 = time.time()
        local = fut.result() if fut is not None else fetch(fn, sizes[fn])   # last shard may not be prefetched
        fut = pool.submit(fetch, todo[j + 1], sizes[todo[j + 1]]) if j + 1 < len(todo) and \
            (pr["bytes"] // 4 + 700_000_000) < target else None
        t1 = time.time()
        st = dict(docs=0, dups=0, tokens=0)
        with open(path, "ab") as f:
            for rb in pq.ParquetFile(local).iter_batches(batch_size=4096, columns=["text"]):
                texts = rb.column(0).to_pylist()
                ids = encode(tok, texts)
                W = np.zeros((len(ids), L), dtype=np.uint64)
                for i, s in enumerate(ids):
                    s = s[:L]; W[i, :len(s)] = s
                hb = hash_rows(W)                        # old_h is sorted (np.unique): binary search, not np.isin
                keep = old_h[np.minimum(np.searchsorted(old_h, hb), len(old_h) - 1)] != hb   # (isin: 18 s/batch)
                st["docs"] += len(ids); st["dups"] += int((~keep).sum())
                out = [np.asarray([BOS] + s, dtype=np.uint32) for s, k in zip(ids, keep) if k and s]
                if out:
                    buf = np.concatenate(out); f.write(buf.tobytes()); st["tokens"] += len(buf)
            f.flush(); os.fsync(f.fileno())
        os.remove(local)
        pr["bytes"] = os.path.getsize(path)
        pr["done_shards"].append(fn); pr["stats"][fn] = st
        save_progress(pr)
        log(f"fwe {fn}: {st['tokens'] / 1e6:,.0f}M tok, {st['docs']:,} docs, {st['dups']:,} dups dropped | "
            f"download {t1 - t0:.0f}s tokenize {time.time() - t1:.0f}s | total {pr['bytes'] / 4e9:.2f}B / {target / 1e9:.1f}B")
    log(f"fwe: done, {pr['bytes'] / 4e9:.3f}B tokens in {len(pr['done_shards'])} shards")


def copy_range(src, dst, off, length, bs=1 << 26):
    with open(src, "rb") as fi:
        fi.seek(off); left = length
        while left > 0:
            b = fi.read(min(bs, left)); dst.write(b); left -= len(b)
            if not b:
                raise IOError(f"short read from {src}")


def phase_assemble():
    n_old = os.path.getsize(OLD) // 4
    split = int(n_old * 0.999)
    fwe = f"{OUT}/fwe_topup.bin"
    n_fwe = os.path.getsize(fwe) // 4
    out = f"{OUT}/train.bin"
    tmp = out + ".tmp"
    log(f"assemble: old train {split:,} + fwe {n_fwe:,} + old val tail {n_old - split:,} tokens")
    with open(tmp, "wb") as f:
        copy_range(OLD, f, 0, split * 4); log("assemble: old train part copied")
        copy_range(fwe, f, 0, n_fwe * 4); log("assemble: fwe copied")
        copy_range(OLD, f, split * 4, (n_old - split) * 4)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, out)
    n_new = os.path.getsize(out) // 4
    assert n_new == n_old + n_fwe
    a = np.memmap(OLD, dtype=np.uint32, mode="r"); b = np.memmap(out, dtype=np.uint32, mode="r")
    assert np.array_equal(a[split:], b[n_new - (n_old - split):]), "val tail mismatch"
    assert np.array_equal(a[:1000], b[:1000])
    old_meta = json.load(open(os.path.join(os.path.dirname(OLD), "meta.json")))
    tr = {k: v["tokens"] for k, v in old_meta["stats"]["train"].items()}
    tr["fineweb_edu"] = tr.get("fineweb_edu", 0) + n_fwe
    tot = sum(tr.values())
    meta = dict(format="flat_token_stream", dtype="uint32", vocab_size=128256, total_tokens=n_new,
                built=time.strftime("%Y-%m-%d %H:%M"), layout="old[:split] + fineweb-edu top-up + old[split:]",
                old_bin=OLD, old_split=split, data_val_tokens=n_old - split,
                fwe_source=f"{REPO} {SUB}", fwe_topup_tokens=n_fwe, progress=load_progress(),
                approx_train_mix={k: round(v / tot, 4) for k, v in sorted(tr.items(), key=lambda x: -x[1])},
                config_keys={"data_bin": out, "data_val_tokens": n_old - split})
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    log(f"assemble: DONE {out} = {n_new / 1e9:.3f}B tokens; config keys: data_bin: {out}  data_val_tokens: {n_old - split}")
    log(f"approx mix: {meta['approx_train_mix']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=8.0e9)
    ap.add_argument("--phase", default="all")
    a = ap.parse_args()
    os.makedirs(f"{OUT}/dl", exist_ok=True)
    if a.phase in ("all", "hashes"):
        phase_hashes()
    if a.phase in ("all", "fwe"):
        phase_fwe(int(a.target))
    if a.phase in ("all", "assemble"):
        phase_assemble()
