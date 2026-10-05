"""check_nl_fact_aux.py -- 2026-10-05. Decode a few NL-fact aux episodes to eyeball them (CPU)."""
import sys
sys.path.insert(0, "/home/glenn/projects/neuro/src"); sys.path.insert(0, "/home/glenn/projects/neuro/scripts")
import yaml
from train_gpu import StreamText, BIN
from nl_fact_aux import NLFactAux
from chat import load_tokenizer
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/r25a_chunk_x1_150m.yaml"))
data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                  val_tokens=c.get("data_val_tokens"))
aux = NLFactAux(c["vocab"], 512, seed=0)
_, dec = load_tokenizer(c["vocab"])
x, last, tgt = aux.batch(6, data, "cpu")
for i in range(6):
    L = int(last[i])
    head = dec(x[i, :60].tolist()).replace("\n", " ")
    tail = dec(x[i, L - 25:L + 1].tolist()).replace("\n", " ")
    print(f"[{i}] last={L} ({'chunk B' if L >= 512 else 'chunk A'}) target={dec([int(tgt[i])])!r}\n   HEAD: {head[:200]}\n   TAIL: ...{tail}")
