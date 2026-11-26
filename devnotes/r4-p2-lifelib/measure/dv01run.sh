#!/bin/bash
# dv01run.sh VARIANT(a|b) MODE(direct|collect) WORKERS -> "variant mode workers seconds cpu peak_mb dv01_total cs01_total"
cd /home/user/ext/lifelib-dv01/lifelib/libraries/uslib/products/fixed_deferred_annuity
flag=""; [ $2 = collect ] && flag="--collect"
out=$(/home/user/ext/venv-new/bin/python run_dv01_sec.py --run /home/user/bench/dv01$1 --input ${BOOK:-/home/user/bench/book1050} --workers $3 --paths 1 --years 2 $flag 2>&1 | tail -1)
echo "$1 $2 $3 $(echo "$out" | python3 -c "import json,sys
try:
    r=json.loads(sys.stdin.read()); print(r['seconds'], r['cpu'], round(r['peak_mb'],1), repr(r['dv01_total']), repr(r['cs01_total']))
except Exception as e: print('FAILED', e)")"
