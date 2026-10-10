"""patch_eval_prelut.py -- 2026-10-10 (Neuro 54). One-shot: the eval scripts built StreamText without data_prelut, so on
a checkpoint trained on the uint16 cloud corpus they read the file as uint32 (garbage ids; screen_p26za_7L_RC3_x50
evals printed nothing). Pass prelut through, default off = unchanged."""
R = "/home/glenn/projects/neuro/"
A = 'val_tokens=c.get("data_val_tokens"))'
B = 'val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))'
for f in ("temp/chunk_eval.py", "temp/xchunk_haystack_eval.py", "temp/nl_fact_eval.py", "temp/xcopy_eval.py",
          "temp/multichunk_eval.py", "temp/xcopy_sweep_eval.py", "temp/energy_eval_v2.py"):
    s = open(R + f).read()
    assert s.count(A) == 1, (f, s.count(A))
    open(R + f, "w").write(s.replace(A, B))
    print("[OK]", f)
f = "temp/haystack_recall_eval.py"
s = open(R + f).read()
a = 'orig_vocab=c.get("data_orig_vocab", 128256))'
assert s.count(a) == 1, (f, s.count(a))
open(R + f, "w").write(s.replace(a, a[:-1] + ', prelut=c.get("data_prelut", False))'))
print("[OK]", f)
