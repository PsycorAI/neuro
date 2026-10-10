#!/bin/bash
# commit_n54.sh -- 2026-10-10 (Neuro 54). One-shot commit of everything (Glenn: "commit all changes").
# Adds: tracked changes; new configs / docs / cloud files; new run logs (not profiler traces, locks, heartbeat);
# new temp scripts (temp/ is git-ignored, existing ones were force-added): *.py *.sh *.ps1 *.list under 1 MB.
cd /home/glenn/projects/neuro
git add -u
git add configs docs cloud
git ls-files --others --exclude-standard logs | grep -E '\.log$' | xargs -r git add
find temp -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.ps1' -o -name '*.list' \) -size -1000k -print0 | xargs -0 -r git add -f
echo "staged: $(git diff --cached --name-only | wc -l) files, $(git diff --cached --shortstat)"
git diff --cached --name-only | awk -F/ '{print $1}' | sort | uniq -c
git diff --cached --name-only --diff-filter=A | while read -r f; do s=$(stat -c %s "$f"); [ "$s" -gt 2000000 ] && echo "BIG $s $f"; done
git commit -q -F temp/commit_n54.msg && git log --oneline | head -n 1
git status --short | grep -v '^??' | head -n 5
