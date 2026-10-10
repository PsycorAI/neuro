#!/bin/bash
cd /home/glenn/projects/neuro
for r in r26m_13L_RC2_hola256_150m r26l_13L_RC2_conv_sc20frz_120M_150m r26n_13L_RC2_xlate_150m r26k_13L_RC2_hola_seed1_150m; do
  echo "== $r"; grep -E "r25-evals $r .*chunk1" logs/gap_queue.log | tail -1 | cut -c40-200; grep -E "$r\] exit" logs/gap_queue.log | tail -1 | cut -c1-160; ls models/checkpoints/$r 2>/dev/null | tr "
" " "; echo; done
