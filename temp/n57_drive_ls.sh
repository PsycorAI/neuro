#!/bin/bash
# Neuro 57: same listing as temp/n56_drive_ls.sh (names in Glenn's public Drive corpus folder, nothing downloaded),
# but the instance address comes from temp/n56_host.env (n56_drive_ls.sh still has the first, outbid instance hard-coded).
. /home/glenn/projects/neuro/temp/n56_host.env
$SSH root@$H 'bash -s' <<'REMOTE' 2>/dev/null
/workspace/venv/bin/python - <<'PYEOF' 2>&1 | tail -n 60
import gdown
fs = gdown.download_folder(url="https://drive.google.com/drive/folders/1ryzuUporjQKZ95v_2sJCI1iEDpozZEbh",
                           skip_download=True, quiet=True)
names = sorted(f.path for f in fs)
for n in names:
    print(n)
print(len(names), "files;", sum("part_" in n for n in names), "corpus pieces")
PYEOF
REMOTE
