"""apply_fpt_nosync.py -- 2026-09-27 (NOT yet applied; needs Glenn's OK -- live post-training imports lif_parallel).
Removes lif_parallel's torch.equal early exit (one blocking host sync per FPT iteration).
Bit-identical: past a fixed point every iteration reproduces s, and at T=512 the loop never
converges anyway (mean iters = K = 10.00 measured), so the check only ever cost syncs.
Measured (5080): 30M eager screen config +20% tok/s (11,058 -> 13,292) with IDENTICAL losses over 125 steps;
v4 draft compiled B8 = no gain (3,415 base vs 3,310; an earlier 2,528 baseline was GPU-contended).
"""
p = "/home/glenn/projects/neuro/src/lif_parallel.py"
s = open(p).read()
old = """    s = torch.zeros_like(cur)
    with torch.no_grad():
        for _ in range(K):
            s_prev = torch.zeros_like(s)
            s_prev[:, 1:, :] = s[:, :-1, :]
            x = cur - beta * vth * s_prev
            pre_mem = _scan_chunked(beta, x)
            s_new = (pre_mem > vth).to(cur.dtype)
            # Fixed point reached: s^(k+1) == s^(k) means every further
            # iteration reproduces the same s, so stopping early is exact
            # (not an approximation), just skips redundant identical work.
            converged = torch.equal(s_new, s)
            s = s_new
            if converged:
                break
"""
new = """    s = torch.zeros_like(cur)
    with torch.no_grad():
        # No early exit (2026-09-27): a torch.equal convergence check forces a host
        # sync per iteration, and at T=512 the loop never converges before K anyway
        # (measured mean iters = K). Past a fixed point every iteration reproduces s,
        # so always running K iterations is bit-identical; dropping the syncs gave
        # +20% tok/s on the 30M eager screens (no gain on the compiled v4 draft).
        for _ in range(K):
            s_prev = torch.zeros_like(s)
            s_prev[:, 1:, :] = s[:, :-1, :]
            x = cur - beta * vth * s_prev
            s = (_scan_chunked(beta, x) > vth).to(cur.dtype)
"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("[OK] lif_parallel early-exit sync removed")
