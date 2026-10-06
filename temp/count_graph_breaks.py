# count_graph_breaks.py -- 2026-09-30 (fused-neuron scoping). torch._dynamo.explain on the v4 draft + lif_exact_ml:
# how many graphs / graph breaks does whole-model compile see, and why. Tracing only (no inductor codegen), B=1.
import sys, yaml, torch
from collections import Counter
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
print("torch", torch.__version__, "| triton_op:", hasattr(torch.library, "triton_op"), "| custom_op:", hasattr(torch.library, "custom_op"))
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/flagship_v4_363m.draft.yaml"))
c.update({"lif_exact": True, "lif_levels": 4, "lif_signed": True, "compile": False})
m = build_model(c).cuda().train()
x = torch.randint(0, c["vocab"], (1, 512), device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    ex = torch._dynamo.explain(m)(x)
print(f"graphs {ex.graph_count} | graph breaks {ex.graph_break_count}")
print(Counter(str(r.reason)[:110] for r in ex.break_reasons).most_common(8))
