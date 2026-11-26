#!/bin/bash
# sec_one.sh MODEL: export the shipped book with collect (live total), build with both compilers
M=$1; B=/home/user/bench/sec/$M; L=/home/user/ext/lifelib/lifelib/libraries/uslib/products
mkdir -p /home/user/bench/sec
rows=$(( $(wc -l < $(ls $L/*/model_point_table_sec.csv | xargs -n1 dirname | while read d; do [ -d $d/$M ] && echo $d/model_point_table_sec.csv; done)) - 1 ))
/home/user/ext/venv-new/bin/python /home/user/bench/prep_sec.py $M $rows $B live > $B.prep.log 2>&1 || { echo "$M prep failed"; exit 1; }
cd $B/src && grep -v "^#" spec_sec.py > spec_b.py && sed "s/from exported import mx_model as m/from ${M}_nomx import mx_model as m/" /home/user/bench/sample_c.py > sample_c.py
for v in a b; do
  cp -r $B/src $B/$v
  pp=/home/user/ext/mxc-c3$v
  (cd $B/$v && PYTHONPATH=$pp /home/user/ext/venv-new/bin/python -m modelx_cython ${M}_nomx --sample sample_c.py --spec spec_b.py > build.log 2>&1)
  echo "$M $v $(ls $B/$v/${M}_nomx_cy/_mx_classes*.so 2>/dev/null | wc -l) $(grep -E '^\S*Error|EXPORTED' $B/$v/build.log | head -2 | tr '\n' ' ')"
done
echo "$M LIVE $(grep LIVE $B.prep.log)"
