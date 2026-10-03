"""Completed recursive hash-prefix children survive their failed parent."""
from test_original_frame_shell import _run


def test_recursive_prefix_retry_preserves_colliding_children_and_growth(tmp_path):
    import modelx as mx
    model = mx.new_model('PrefixRetry')
    space = model.new_space('S')
    space.counts = {}
    space.flags = {'fail': False}
    space.width = 30
    space.new_cells('value', '''def value(x):
 counts[x] = counts.get(x, 0) + 1
 if x == 0.0:
  total = 0.0
  for i in range(1, width + 1):
   total += value(float(i))
  if flags['fail']: raise ValueError("retry")
  return total
 return x - 1.0
''')
    model.export(tmp_path/'model')
    model.close()
    (tmp_path/'sample.py').write_text('from model import mx_model\nmx_model.S.value(0.0)\n')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path/'model'),
                    '--sample', str(tmp_path/'sample.py'), '--no-spec'], 'build')
    (tmp_path/'verify.py').write_text('''from model_cy import mx_model
for width in (30, 48):
 m=type(mx_model)();s=m.S;s.width=width;s.flags['fail']=True
 for attempt in range(2):
  try:s.value(0.0)
  except ValueError as error:assert str(error)=='retry'
  else:raise AssertionError('failed parent published a result')
  assert s.counts[0.0]==attempt+1
  for i in range(1,width+1):
   assert s.value(float(i))==i-1.0
   assert s.counts[float(i)]==1, (width,i,s.counts)
 s.flags['fail']=False
 assert s.value(0.0)==width*(width-1)/2
 assert s.value(0.0)==width*(width-1)/2
 assert s.counts[0.0]==3
 assert all(s.counts[float(i)]==1 for i in range(1,width+1))
''')
    _run(tmp_path, ['verify.py'], 'verify')
