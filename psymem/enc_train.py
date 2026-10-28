# psymem/enc_train.py -- 2026-10-27 (Neuro 85): encoder stage 1 -- distil bge-small into the spiking front end (encoder.py).
# loss = (1 - cos(student, teacher)) + lam * KL(teacher || student) over in-batch similarity distributions (temperature tau),
# so the student copies both the vectors and the neighbourhood structure retrieval depends on.
# Data: data/psymem_enc/corpus_N.jsonl + emb_N.npy (bge on raw turns); the last 5k turns are held out:
#   val cos, and nbr@10 = share of the teacher's 10 nearest held-out turns that the student also ranks in its top 10.
# Usage: python psymem/enc_train.py TAG N EPOCHS [d layers]   -> models/psymem/enc_TAG.pt
import sys, json, time, math, random
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer
import encoder
from spike_code import SpikeCode

D = '/home/glenn/projects/neuro/'
B, LR, WARM, TAU, LAM, NV = 256, 2e-3, 500, 0.05, 1.0, 5000


def batches(idx, ids, rng):  # length-bucketed random batches
    idx = list(idx); rng.shuffle(idx)
    out = []
    for i in range(0, len(idx), 50 * B):
        mb = sorted(idx[i:i + 50 * B], key=lambda j: len(ids[j]))
        out += [mb[j:j + B] for j in range(0, len(mb), B)]
    rng.shuffle(out)
    return out


def collate(b, ids):
    L = max(len(ids[j]) for j in b)
    x = torch.zeros(len(b), L, dtype=torch.long); m = torch.zeros(len(b), L, dtype=torch.bool)
    for r, j in enumerate(b):
        x[r, :len(ids[j])] = torch.from_numpy(ids[j]); m[r, :len(ids[j])] = True
    return x.cuda(non_blocking=True), m.cuda(non_blocking=True)


@torch.no_grad()
def embed(model, idx, ids):
    model.eval()
    out = torch.cat([model(*collate(b, ids)) for b in [idx[i:i + 512] for i in range(0, len(idx), 512)]])
    model.train()
    return out


if __name__ == '__main__':
    tag, n, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cfg = dict(d=int(sys.argv[4]) if len(sys.argv) > 4 else 384, layers=int(sys.argv[5]) if len(sys.argv) > 5 else 4)
    texts = [json.loads(l) for l in open(D + f'data/psymem_enc/corpus_{n}.jsonl')]
    T = torch.tensor(np.load(D + f'data/psymem_enc/emb_{n}.npy'), dtype=torch.float32).cuda()
    tok, ids = AutoTokenizer.from_pretrained(encoder.TOK), []
    for i in range(0, n, 20000):  # in chunks, kept as int32 arrays (one call on all 400k held ~18 GB of host RAM)
        ids += [np.array(x, np.int32) for x in tok(texts[i:i + 20000], truncation=True, max_length=256)['input_ids']]
    del texts
    tr, va = list(range(n - NV)), list(range(n - NV, n))
    torch.manual_seed(0); rng = random.Random(0)
    model = encoder.SpikeEnc(**cfg).cuda()
    print(f'{tag}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, cfg {cfg}', flush=True)
    codes = [m for m in model.modules() if isinstance(m, SpikeCode)]
    for m in codes:  # calibrate spike thresholds on real activations (coding bypassed)
        m.calib = []
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
        for b in batches(tr, ids, rng)[:8]:
            model(*collate(b, ids))
    for m in codes:
        m.theta.data.copy_(torch.stack(m.calib).mean(0).clamp(min=1e-3)); m.calib = None
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps, step, t0 = epochs * (len(tr) // B), 0, time.time()
    tv = T[va]; tnb = (tv @ tv.T).fill_diagonal_(-2).topk(10, dim=1).indices
    for ep in range(epochs):
        for b in batches(tr, ids, rng):
            lr = LR * min(1, (step + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(1, step / steps)))
            for g in opt.param_groups:
                g['lr'] = lr
            with torch.autocast('cuda', torch.bfloat16):
                s = model(*collate(b, ids))
            t = T[b]
            eye = torch.eye(len(b), device='cuda', dtype=torch.bool)
            ls = F.log_softmax((s @ s.T / TAU).masked_fill(eye, -1e4), 1)
            pt = F.softmax((t @ t.T / TAU).masked_fill(eye, -1e4), 1)
            l_cos = (1 - (s * t).sum(1)).mean()
            l_kl = (pt * (torch.log(pt + 1e-9) - ls)).sum(1).mean()
            loss = l_cos + LAM * l_kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            if step % 1000 == 0 or step == steps:
                sv = embed(model, va, ids)
                snb = (sv @ sv.T).fill_diagonal_(-2).topk(10, dim=1).indices
                nbr = (snb[:, :, None] == tnb[:, None, :]).any(2).float().mean().item()
                spk = np.mean([m.last_nz.item() for m in codes])
                print(f'step {step}/{steps} loss {loss.item():.4f} cos {l_cos.item():.4f} kl {l_kl.item():.4f} | val cos '
                      f'{(sv * tv).sum(1).mean().item():.4f} nbr@10 {nbr:.3f} | spikes nz {spk:.2f} lr {lr:.2e} '
                      f'{time.time() - t0:.0f}s', flush=True)
                torch.save(dict(cfg=cfg, model=model.state_dict(), step=step), D + f'models/psymem/enc_{tag}.pt')
    print('saved', D + f'models/psymem/enc_{tag}.pt')
