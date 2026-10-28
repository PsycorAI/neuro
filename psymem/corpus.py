# psymem/corpus.py -- 2026-10-27 (Neuro 85): unlabelled chat turns for the Psymem encoder (Hebbian step 0, distillation,
# question generation). Source: allenai/tulu-3-sft-mixture (HF cache). No LoCoMo / LongMemEval text (those stay test sets).
# Usage: python psymem/corpus.py N   -> data/psymem_enc/corpus_N.jsonl (texts, "role: content") + emb_N.npy (bge, fp16)
import os, sys, json, random
os.environ['HF_DATASETS_OFFLINE'] = '1'
import numpy as np
from datasets import load_dataset
import bench

OUT = '/home/glenn/projects/neuro/data/psymem_enc/'

if __name__ == '__main__':
    N = int(sys.argv[1])
    os.makedirs(OUT, exist_ok=True)
    ds = load_dataset('allenai/tulu-3-sft-mixture', split='train')
    order = list(range(len(ds))); random.Random(0).shuffle(order)
    texts, seen = [], set()
    for i in order:
        for msg in ds[i]['messages']:
            t = msg['content'].strip()
            if msg['role'] in ('user', 'assistant') and 30 <= len(t) <= 3000 and t not in seen:
                seen.add(t); texts.append(f"{msg['role']}: {t}")
        if len(texts) >= N:
            break
    texts = texts[:N]
    with open(OUT + f'corpus_{N}.jsonl', 'w') as f:
        for t in texts:
            f.write(json.dumps(t) + '\n')
    emb = bench.Embedder()
    E = np.concatenate([emb(texts[i:i + 20000]).astype(np.float16) for i in range(0, N, 20000)])
    np.save(OUT + f'emb_{N}.npy', E)
    print(f'{N} turns from {len(ds)} conversations; emb {E.shape}', flush=True)
