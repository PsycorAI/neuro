#!/bin/bash
# hold_sepbw.sh -- 2026-10-11 (Neuro 55): hold r26x / r26y so the 5090 chain exits after r26w, write the conversion
# smoke config with mqar_separate_backward, and start the validation driver (temp/queue_5090_sepbw.sh), which
# restores the list and re-runs the chain when it is done.
cd /home/glenn/projects/neuro
pgrep -f "^bash temp/queue_5090_sepbw.sh" > /dev/null && { echo "driver already running"; exit 0; }
rm -f temp/sepbw.READY
sed -i -E 's/^(r26[xy]_)/#HOLD \1/' temp/queue_5090_n54.list
/home/glenn/projects/bdh/venv/bin/python - <<'EOF'
import yaml
R = "/home/glenn/projects/neuro/configs/"
c = yaml.safe_load(open(R + "v4_smoke1b_conv.yaml"))
c.update(run_name="v4_smoke1b_conv_sepbw", mqar_separate_backward=True)
with open(R + "v4_smoke1b_conv_sepbw.yaml", "w") as f:
    f.write("# v4_smoke1b_conv_sepbw -- 2026-10-11 (temp/hold_sepbw.sh). The 1B conversion smoke at the real shape + mqar_separate_backward.\n")
    yaml.safe_dump(c, f, sort_keys=False)
print("wrote v4_smoke1b_conv_sepbw")
EOF
bash -n temp/queue_5090_sepbw.sh || exit 1
setsid -f bash temp/queue_5090_sepbw.sh < /dev/null > /dev/null 2>&1
sleep 1; tail -n 4 temp/queue_5090_n54.list; pgrep -af "^bash temp/queue_5090" | cut -c1-90
grep -a '^step' logs/r26w_13L_RC3_headwd03_150m.log | tail -n 1 | cut -c1-70
