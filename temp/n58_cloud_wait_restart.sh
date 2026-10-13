#!/bin/bash
# Neuro 58: after n58_cloud_apply.sh apply -- poll the instance (max $1 min, default 9) until the restarted trainer has
# logged 3 step lines after its new "resumed" line, then print the verify view.
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
end=$(( $(date +%s) + ${1:-9} * 60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  n=$($SSH root@$H 'cd /workspace/neuro; awk "/^resumed /{r++; n=0} /^step [0-9]+ \|/{n++} END{print r+0, n+0}" logs/v4_cloud.log' 2>/dev/null)
  echo "$(date +%H:%M:%S) resumes / steps since the last resume: $n"
  set -- $n; [ "${1:-0}" -ge 1 ] && [ "${2:-0}" -ge 3 ] && [ "${2:-0}" -lt 200 ] && break
  sleep 30
done
bash temp/n58_cloud_apply.sh verify
