"""Where does the corpus-v4 builder spend its time per 4096-doc batch? (read / to_pylist / encode / numpy)"""
import sys, time, os, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/home/glenn/projects/neuro/scripts")
import build_corpus_v4 as B
tok = B.load_tok()
t = time.time(); pf = pq.ParquetFile("/mnt/g/tmp/fwe_v4/000_00000.parquet"); it = pf.iter_batches(batch_size=4096, columns=["text"])
rb = next(it); t_read = time.time() - t
t = time.time(); texts = rb.column(0).to_pylist(); t_list = time.time() - t
t = time.time(); ids = B.encode(tok, texts); t_enc = time.time() - t
n = sum(len(x) for x in ids)
t = time.time()
W = np.zeros((len(ids), B.L), dtype=np.uint64)
for i, s in enumerate(ids):
    s = s[:B.L]; W[i, :len(s)] = s
h = B.hash_rows(W)
out = np.concatenate([np.asarray([B.BOS] + s, dtype=np.uint32) for s in ids if s])
t_np = time.time() - t
t = time.time(); rb2 = next(it); t_read2 = time.time() - t
print(f"tokens {n:,} | read {t_read:.2f}s (next batch {t_read2:.2f}s) | to_pylist {t_list:.2f}s | encode {t_enc:.2f}s "
      f"({n / t_enc / 1e6:.2f}M tok/s) | numpy {t_np:.2f}s | RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')}")
