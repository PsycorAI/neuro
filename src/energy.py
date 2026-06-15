"""Inference-energy accounting (Goal 2), hardware-independent.

We use the standard 45nm energy-per-operation model (Horowitz, ISSCC 2014) that
SNN papers (SpikeGPT, SpikingBrain) use:
    MAC (multiply-accumulate, dense float)  ~ 4.6 pJ
    AC  (accumulate, triggered by a spike)  ~ 0.9 pJ   (~5x cheaper)

Spiking layers turn dense MACs into sparse ACs: a synapse only does work when its
presynaptic neuron spikes. The Hebbian memory is recurrent, so its cost per token is
O(d_mem^2) and does NOT grow with context length -- unlike transformer attention,
whose per-token cost grows with the number of past tokens attended to.

CAVEAT: this is the energy a neuromorphic / event-driven chip would use. A dense GPU
(your 5080/5090) cannot exploit spike sparsity and will not realize these savings at
inference; it may even be slower. The number measures the COMPUTATION, not the GPU.
"""

E_MAC = 4.6e-12   # joules
E_AC = 0.9e-12    # joules


def spiking_energy_per_token(model, spike_rate):
    d, N, dm, V = model.d, model.n_neurons, model.d_mem, model.vocab
    mac = d * N            # to_current: continuous embedding -> current (dense MAC)
    mac += 2 * dm * dm     # Hebbian write (v k^T) + read (M q)
    mac += dm * V          # output head
    ac = spike_rate * N * dm * 3   # W_k, W_v, W_q are driven by binary spikes -> AC
    energy = mac * E_MAC + ac * E_AC
    return energy, mac, ac


def transformer_energy_per_token(d, n_layer, vocab, seq_len):
    # autoregressive inference: each new token attends to `seq_len` past tokens
    per_layer = 12 * d * d + 2 * d * seq_len   # qkv+out+mlp (12 d^2) + attention (2 d L)
    mac = n_layer * per_layer + d * vocab
    return mac * E_MAC, mac


def compare(model, spike_rate, baseline_d, baseline_layers, vocab, seq_lens):
    e_s, mac_s, ac_s = spiking_energy_per_token(model, spike_rate)
    print(f"{'seq_len':>8} | {'spiking pJ/tok':>15} | {'transf. pJ/tok':>15} | {'ratio (T/S)':>12}")
    print("-" * 60)
    for L in seq_lens:
        e_t, _ = transformer_energy_per_token(baseline_d, baseline_layers, vocab, L)
        print(f"{L:>8} | {e_s*1e12:>15.1f} | {e_t*1e12:>15.1f} | {e_t/e_s:>12.2f}x")
    print(f"\nspiking op mix/token: {mac_s:,} MAC + {ac_s:,.0f} AC (spike_rate={spike_rate:.3f})")
    print("note: spiking cost is constant in seq_len; transformer grows with it.")
