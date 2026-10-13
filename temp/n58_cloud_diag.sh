#!/bin/bash
# Neuro 58: read-only. The cloud run's last evals with all telemetry lines, mean train loss per 1,000 steps, tripwire lines.
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
$SSH root@$H 'cd /workspace/neuro; L=logs/v4_cloud.log
echo "== mean train loss per 1000 steps (last 14)"
grep -E "^step [0-9]+ " $L | awk "{s=int(\$2/1000); sum[s]+=\$8; n[s]++} END{for(k in n) printf \"%d %.3f %d\n\", k*1000, sum[k]/n[k], n[k]}" | sort -n | tail -n 14 | tr "\n" ";"
echo; echo "== last 2 eval blocks"
awk "/\[eval\]/{c++} {a[NR]=\$0; e[NR]=c} END{for(i=1;i<=NR;i++) if (e[i]>=c-1 && a[i] !~ /^step/) print a[i]}" $L | cut -c1-330 | tail -n 40
echo "== tripwire / watchdog / resume"
grep -n -i -E "tripwire|watchdog|resumed|rollback|nan" $L | tail -n 6 | cut -c1-200
tail -n 4 logs/run_v4.log | cut -c1-200
date -u "+%m-%d %H:%M UTC"; grep -E "^step" $L | tail -n 1 | cut -c1-100' 2>/dev/null
