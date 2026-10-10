#!/bin/bash
cd /home/glenn/projects/neuro
for d in r26o_13L_RC3_conv_llamakd_150m r26k_13L_RC2_hola_seed1_150m r26m_13L_RC2_hola256_150m; do echo "== $d: $(ls models/checkpoints/$d | tr "
" " ")"; grep "^step" logs/$d.log | tail -1; grep -c "^done:" logs/$d.log; done
df -h / | tail -1
