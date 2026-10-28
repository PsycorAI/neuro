# Memory products: published scores vs Plasmem (Neuro 85, 2026-10-27)

Glenn chose option (a): compare against published scores, and disclose that the setups differ. A local head-to-head with one
shared reader model (about 14B, run on the 5090) comes later, before any public claim.

## Published LoCoMo scores

Score = an LLM judge's accuracy on the final answer. Readers and judges are GPT-4-class unless noted. Most numbers are
reported by the vendors themselves.

| System | LoCoMo (judge) | Source / caveat |
|---|---|---|
| Zep | 94.7 (vendor, 2026); independent runs ~58–75 | Cited in Mem0's blog; braindetox.kr shows 58–95 for the same product |
| Mem0 | 91.6–92.5 (vendor, 2026); 66.9 (Mem0 paper, Apr 2025); 68.5 (third party) | mem0.ai blog posts |
| Letta | 74.0 (filesystem + grep); 83.2 (another write-up) | Towards AI; digitalapplied.com |
| LangMem | 58.1 / 78.1 (two runs) | youngju.dev; Memori benchmark page |
| **Plasmem v0** (ours) | **64.1** with Llama-3.2-3B reader + Phi-3.5 judge, k 10; hybrid_cc_nb 63.4; RRF 56.6 | logs/n85_qa_locomo.log; local small models, so not comparable in absolute terms |

## Why the numbers are not comparable

- Judge accuracy depends mostly on the reader and the judge (GPT-4o vs a 3B model) and on the prompts. The same product
  scores anywhere from 58 to 95 depending on who runs it.
- Many vendors drop category 5 (adversarial) questions or score categories differently.
- A fair comparison would hold the reader, the judge and the prompts fixed and swap only the memory. Our recall@k numbers
  do exactly that against strong ordinary search.

## Where Plasmem can honestly win now (features, not judge scores)

- **No LLM call to write a memory.** Mem0 and Zep run an LLM on every write (extraction, graph updates); Plasmem's write is
  a Hebbian update in microseconds and costs nothing. This is a cost, latency and privacy advantage.
- **Learns from use:** paraphrased recall goes 57 → 89 after one correction. We know of no published equivalent.
- **Explainable recall:** every result shows which neurons and synapses fired.
- **One local file you own:** CPU only, about 50 ms per recall on a small diary.
- **Time cells:** built-in understanding of time expressions.

## To do

1. Local head-to-head: one 14B reader and judge, Mem0 open source + LangMem + plain RAG + Plasmem, same prompts. Status:
   to-test.
2. Measure write cost and latency for each system. Status: to-test.

Sources: [mem0.ai: Zep vs Mem0](https://mem0.ai/blog/zep-vs-mem0-which-ai-memory-layer-should-you-choose) ·
[mem0.ai: AI memory benchmarks](https://mem0.ai/blog/ai-memory-benchmarks) ·
[braindetox.kr: one product, 58 to 95 on LoCoMo](https://braindetox.kr/en/posts/memory_benchmark_scores_reliability_2026.html) ·
[Towards AI: Mem0 vs Zep vs Letta](https://pub.towardsai.net/mem0-vs-zep-vs-letta-a-folder-of-text-files-shouldnt-beat-the-61k-star-memory-layer-9d7e65c5799c) ·
[digitalapplied.com](https://www.digitalapplied.com/blog/open-source-agent-memory-mem0-letta-zep-compared) ·
[Memori: LoCoMo results](https://memorilabs.ai/docs/memori-cloud/benchmark/results/) ·
[youngju.dev](https://youngju.dev/blog/2026-07-17-agent-memory-what-is-actually-measured)
