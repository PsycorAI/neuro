#!/bin/bash
# list uncommitted files by size (KB), largest first
cd /home/glenn/projects/neuro
git status --porcelain --untracked-files=all | cut -c4- | while read -r f; do du -k "$f" 2>/dev/null; done | sort -rn | head -20
