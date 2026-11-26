#!/bin/bash
# sec_bench.sh ROUNDS MODELS... : kept and collect with the borrowed read (b), collect without (a), kept without on 3 models
R=$1; shift
for r in $(seq 1 $R); do for M in "$@"; do for w in 1 4; do
  vs="b:kept b:thin a:thin"; case $M in Term_US_SEC|MYGA_US_SEC|VA_US_SEC) vs="$vs a:kept";; esac
  for v in $vs; do IFS=: read c mode <<< "$v"
    (cd /home/user/bench/sec/$M/$c && echo "$M $r $c $(PYTHONPATH=. /home/user/ext/venv-new/bin/python /home/user/bench/run.py ${M}_nomx_cy $mode $w 2>&1 | tail -1)")
  done; done; done; done
