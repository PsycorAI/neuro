p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "Neuro 46 queued a ~2h 1B neuron probe A/B after the chain (temp/run_probe_neuron_ab.sh); Neuro 45 owns the 5080 queue."
assert s.count(old) == 1
new = old + (" <b style=\"color:#c0392b\">POST-TRAINING OUTCOME (10-02 10:48):</b> ORPO COLLAPSED &mdash; word soup (&lsquo;beef, beef, beef&hellip;&rsquo;), recall 84/72/28/6/0, tool_eval mode .39 / tool 0 / free-mode parse 0, refuses everything; chosen&gt;rejected drifted 81%&rarr;44%. "
 "Config had lr 1e-4, orpo_lambda 0.1, NO MQAR rehearsal. OPD (on ORPO, 14.7h) inherited it; head_energy done; head_conf killed (exit 143 SIGTERM, cause unknown; degenerate labels). "
 "<b>RELEASE/DEMO CANDIDATE = sft_v3_363m/step_0003663.</b> TO-DECIDE (Glenn): (a) ORPO retry with MQAR rehearsal + lr ~2e-5 + short run + generation-check early stop, or (b) skip preference, run OPD from SFT (opd_init: sft_v3_363m, ~15h), or (c) ship SFT for the demo. 4th preference-stage collapse at this scale.")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
