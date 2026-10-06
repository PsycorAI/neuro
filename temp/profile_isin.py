import time, numpy as np
old = np.load("/home/glenn/projects/bdh/data/tokenized_v4/old_doc_hashes.npy")
h = np.random.randint(0, 2**63, size=4096, dtype=np.int64).astype(np.uint64)
t = time.time(); a = np.isin(h, old); t1 = time.time() - t
t = time.time()
idx = np.searchsorted(old, h); b = old[np.minimum(idx, len(old) - 1)] == h
t2 = time.time() - t
print(f"np.isin {t1:.2f}s  searchsorted {t2 * 1000:.2f}ms  same={np.array_equal(a, b)}  sorted={bool(np.all(old[1:] > old[:-1]))}")
