# psymem/enc_tune.py -- 2026-10-27 (Neuro 85): encoder stage 2 -- contrastive memory-retrieval fine-tuning of the spiking
# front end on synthetic (question, turn) pairs (psymem/qgen.py), with the stage-1 distillation loss kept as a regularizer
# (one pair batch + one corpus batch per step). Pair loss = symmetric InfoNCE over the batch (other turns = negatives).
# Hard negatives (10-27): pairs with "negs" (SODA conversation pairs) add one other turn of the same conversation to the
# candidate set, so the encoder must tell apart turns of one chat (as in LoCoMo), not just topics.
# Val: 2,000 held-out pairs, r@1 / r@10 of each question's own turn among all 2,000 (bge-small with its query prefix printed
# as the reference). No LoCoMo / LongMemEval data.
# Usage: python psymem/enc_tune.py TAG_IN TAG_OUT EPOCHS   (reads models/psymem/enc_TAG_IN.pt, writes enc_TAG_OUT.pt)
import sys, json, time, math, random, glob
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer
import encoder, bench
from enc_train import batches, collate, embed, D, B, TAU, NV

LR, WARM, NVP = 5e-4, 200, 2000


def toks(tok, texts):
    out = []
    for i in range(0, len(texts), 20000):
        out += [np.array(x, np.int32) for x in tok(texts[i:i + 20000], truncation=True, max_length=256)['input_ids']]
    return out


def retrieval(qv, pv):  # r@1, r@10 of q_i -> p_i among all p
    rank = ((qv @ pv.T) > (qv * pv).sum(1, keepdim=True)).sum(1)
    return (rank < 1).float().mean().item(), (rank < 10).float().mean().item()


if __name__ == '__main__':
    tin, tout, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    pairs = [json.loads(l) for f in sorted(glob.glob(D + 'data/psymem_enc/pairs_*.jsonl')) for l in open(f)
             if l.endswith('\n')]  # skip a half-written last line (a generator may still be appending)
    random.Random(0).shuffle(pairs)
    tok = AutoTokenizer.from_pretrained(encoder.TOK)
    qi, pi = toks(tok, [p['q'] for p in pairs]), toks(tok, [p['pos'] for p in pairs])
    nr = random.Random(5)
    ni = toks(tok, [nr.choice(p['negs']) if p.get('negs') else '' for p in pairs])
    texts = [json.loads(l) for l in open(D + 'data/psymem_enc/corpus_400000.jsonl')]
    T = torch.tensor(np.load(D + 'data/psymem_enc/emb_400000.npy'), dtype=torch.float32).cuda()
    ci = toks(tok, texts); n = len(texts); del texts
    ck = torch.load(D + f'models/psymem/enc_{tin}.pt', map_location='cpu')
    model = encoder.SpikeEnc(**ck['cfg']).cuda(); model.load_state_dict(ck['model'])
    va, tr = list(range(NVP)), list(range(NVP, len(pairs)))
    te = bench.Embedder()
    tq = torch.tensor(te([bench.Embedder.QPFX + pairs[i]['q'] for i in va])).cuda()
    tp = torch.tensor(te([pairs[i]['pos'] for i in va])).cuda()
    kinds = {k: sum(p['kind'] == k for p in pairs) for k in ('recall', 'personal', 'conv')}
    print(f'{tout}: {len(tr)} train pairs {kinds}, val {NVP}; bge-small reference r@1/r@10 {retrieval(tq, tp)}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(1)
    steps, step, t0 = epochs * (len(tr) // B), 0, time.time()

    def val():
        r = retrieval(embed(model, va, qi), embed(model, va, pi))
        sv, tv = embed(model, list(range(n - NV, n)), ci), T[n - NV:]
        return r, (sv * tv).sum(1).mean().item()
    print(f'step 0 val r@1/r@10 {val()[0]}', flush=True)
    for ep in range(epochs):
        cb = batches(range(n - NV), ci, rng)
        for k, b in enumerate(batches(tr, pi, rng)):
            lr = LR * min(1, (step + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(1, step / steps)))
            for g in opt.param_groups:
                g['lr'] = lr
            # two backward passes (gradients add up = same as one on the sum): only one graph lives at a time, since three
            # B=256 forwards at once ran the 5090 out of VRAM ("device not ready", 10-27)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', torch.bfloat16):
                c = model(*collate(cb[k % len(cb)], ci))
            l_dist = (1 - (c * T[cb[k % len(cb)]]).sum(1)).mean()
            l_dist.backward(); del c
            with torch.autocast('cuda', torch.bfloat16):
                q, p = model(*collate(b, qi)), model(*collate(b, pi))
                hb = [j for j in b if pairs[j].get('negs')]
                if hb:
                    p = torch.cat([p, model(*collate(hb, ni))])
            S = q @ p.T / TAU
            lab = torch.arange(len(b), device='cuda')
            l_pair = (F.cross_entropy(S, lab) + F.cross_entropy(S[:, :len(b)].T, lab)) / 2
            l_pair.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            if step % 500 == 0 or step == steps:
                (r1, r10), vc = val()
                print(f'step {step}/{steps} pair {l_pair.item():.4f} dist {l_dist.item():.4f} | val r@1 {r1:.3f} r@10 {r10:.3f} '
                      f'corpus cos {vc:.4f} | lr {lr:.2e} {time.time() - t0:.0f}s', flush=True)
                torch.save(dict(cfg=ck['cfg'], model=model.state_dict(), step=step), D + f'models/psymem/enc_{tout}.pt')
    print('saved', D + f'models/psymem/enc_{tout}.pt')
