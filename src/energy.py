"""Inference-energy accounting (Goal 2), hardware-independent.

We use the standard 45nm energy-per-operation model (Horowitz, ISSCC 2014) that
SNN papers (SpikeGPT, SpikingBrain) use:
    MAC (multiply-accumulate, dense float)  ~ 4.6 pJ
    AC  (accumulate, triggered by a spike)  ~ 0.9 pJ   (~5x cheaper)

Spiking layers turn dense MACs into sparse ACs: a synapse only does work when its
presynaptic neuron spikes.

CAVEAT: this is the energy a neuromorphic / event-driven chip would use. A dense GPU
(your 5080/5090) cannot exploit spike sparsity and will not realize these savings at
inference; it may even be slower. The number measures the COMPUTATION, not the GPU.

2026-08-08 rewrite (docs/pre_flagship_audit_2026_08_08.md B2). The previous
version predated both GDN-2 and hybrid attention and got three things wrong,
all of which inflated the reported advantage:

  1. HybridAttentionBlock was not counted AT ALL. It is a dense transformer
     layer: 12*d^2 (qkv + o_proj + FFN) plus 2*d*seq_len for the attention
     itself. Its cost grows with context exactly like the baseline's, so the
     old closing claim "spiking cost is constant in seq_len" is false for any
     hybrid model.
  2. The AC term counted 3 spike-driven projections per block. GDN-2 made it
     7 (W_k, W_v, W_q, W_ff + the erase/write/decay gates W_b, W_w, W_fdecay).
  3. n_layers multiplied the spiking terms even though attention layers emit
     no spikes at all.

It also now corrects the spike-rate dilution (gotcha #5) internally: the rate
handed in by train_gpu.evaluate() is a flat mean over ALL blocks including the
attention layers' zero stubs, so it under-reports the true spiking-layer rate
by n_spiking/n_total. Energy must use the true rate.
"""

E_MAC = 4.6e-12   # joules
E_AC = 0.9e-12    # joules


def _block_mix(model):
    """(n_spiking, n_attention) over the real block stack."""
    n_attn = sum(1 for b in model.blocks if hasattr(b, "qkv"))
    return len(model.blocks) - n_attn, n_attn


def spiking_energy_per_token(model, spike_rate, seq_len):
    """Energy for ONE token at inference, walking the actual block stack.

    `spike_rate` is the DILUTED mean as reported by train_gpu.evaluate()
    (flat mean over all blocks, attention layers contributing 0); it is
    rescaled to the true spiking-layer rate here.

    Excludes the MTP head -- it is a training auxiliary and only runs at
    inference under speculative decoding.
    """
    d, N, dm, V = model.d, model.n_neurons, model.d_mem, model.vocab
    n_spiking, n_attn = _block_mix(model)
    n_total = n_spiking + n_attn
    rate = spike_rate * n_total / n_spiking if n_spiking else 0.0

    mac = ac = 0.0
    for i, b in enumerate(model.blocks):
        if hasattr(b, "qkv"):                       # HybridAttentionBlock
            # qkv (3d^2) + o_proj (d^2) + FFN (8d^2, same for SwiGLU or 4d GELU)
            mac += 12 * dm * dm
            # autoregressive attention: this token against seq_len past keys,
            # then the same against values. GROWS with context.
            mac += 2 * dm * seq_len
            continue
        d_in = d if i == 0 else dm
        mac += d_in * N                             # to_current (dense residual in)
        mac += 3 * dm * dm                          # memory: decay-read, erase-read, query-read
        # Spike-driven projections: k/v/q/ff always, + GDN-2's three gates.
        n_proj = 4 + (3 if getattr(b, "gdn2_enabled", False) else 0)
        ac += rate * n_proj * N * dm
    mac += dm * V                                   # output head
    return mac * E_MAC + ac * E_AC, mac, ac, rate


def transformer_energy_per_token(d, n_layer, vocab, seq_len):
    # autoregressive inference: each new token attends to `seq_len` past tokens
    per_layer = 12 * d * d + 2 * d * seq_len   # qkv+out+mlp (12 d^2) + attention (2 d L)
    mac = n_layer * per_layer + d * vocab
    return mac * E_MAC, mac


def compare(model, spike_rate, baseline_d, baseline_layers, vocab, seq_lens):
    n_spiking, n_attn = _block_mix(model)
    print(f"{'seq_len':>8} | {'spiking pJ/tok':>15} | {'transf. pJ/tok':>15} | {'ratio (T/S)':>12}")
    print("-" * 60)
    last = None
    for L in sorted(set(seq_lens)):
        e_s, mac_s, ac_s, rate = spiking_energy_per_token(model, spike_rate, L)
        e_t, _ = transformer_energy_per_token(baseline_d, baseline_layers, vocab, L)
        print(f"{L:>8} | {e_s*1e12:>15.1f} | {e_t*1e12:>15.1f} | {e_t/e_s:>12.2f}x")
        last = (L, mac_s, ac_s, rate)
    L, mac_s, ac_s, rate = last
    print(f"\nblock mix: {n_spiking} spiking + {n_attn} attention "
          f"| spike_rate {spike_rate:.3f} reported -> {rate:.3f} spiking-only")
    print(f"spiking op mix/token @ seq {L}: {mac_s:,.0f} MAC + {ac_s:,.0f} AC")
    if n_attn:
        print(f"note: {n_attn} full-attention layer(s) mean the spiking model's cost "
              f"ALSO grows with seq_len,\n      just far more slowly than the "
              f"all-attention baseline's.")
    else:
        print("note: spiking cost is constant in seq_len; transformer grows with it.")
