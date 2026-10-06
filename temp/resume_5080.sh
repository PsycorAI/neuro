#!/bin/bash
# resume_5080.sh -- resume the 5080 screens after pause_5080.sh. Each runner skips
# finished arms (.done markers) and each arm auto-resumes from its latest ckpt:
# independent-windows screen first, then the depth screen.
cd /home/glenn/projects/neuro
bash temp/run_5080_indepwin.sh
bash temp/run_5080_depth.sh
