#!/bin/bash
# git_untracked_sizes.sh -- list untracked files (not ignored) by top directory, and any above 1 MB.
cd /home/glenn/projects/neuro
git ls-files --others --exclude-standard | awk -F/ '{print $1"/"($2 ~ /\./ ? "" : $2)}' | sort | uniq -c | sort -rn | head -n 20
echo "--- above 1 MB:"
git ls-files --others --exclude-standard | while read -r f; do s=$(stat -c %s "$f"); [ "$s" -gt 1000000 ] && echo "$s $f"; done
echo "--- logs:"
git ls-files --others --exclude-standard logs | head -n 40
