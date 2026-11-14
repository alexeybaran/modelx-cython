"""A NativeTable's index answers as the live helper's index (D54).

Compiled Endowment_JP_S read ``sid in tbl.index`` (a level-0 label of a two-level table,
held in a local, so not lowered) through ``_NativeTableIndex.__contains__``, which kept the
round-2 full-key test and answered False where the helper tests the prefix. Every member
of the index view now gives the helper's answer; this test reads each of them live,
exported and compiled, through a local that the lowering leaves to Python, and through the
lowered ``has_1`` path.
"""

import json
import subprocess
import sys

import modelx as mx
import pandas as pd
from modelx_helpers import pdn

PROBE = '''\
def probe(p):
    sid = sid_of(p)
    tbl = data.table()
    index = tbl.index
    return (sid in index, (sid, 3) in index, (sid, 4) in index, sid not in index,
            len(index), list(index), index.min(), index.max(), index[1], index[-1],
            type(tbl).__name__ in ("Table", "NativeTable"))
'''
LOWERED = '''\
def lowered(p):
    return (p + 1) in data.numbered().index
'''
RUN = ('import json\nfrom {package} import mx_model as m\n'
       'print(json.dumps([[repr(m.Projection.probe(p)), m.Projection.lowered(p)] for p in range(4)]))\n'
       'print(type(m.Data.table()).__name__)\n')


def _model(root):
    model = mx.new_model('IndexSemantics')
    try:
        data = model.new_space('Data', refs={'pdn': pdn})
        data.src_frame = pd.DataFrame(
            {'v': [0.05, 0.05, 0.1, 0.2]},
            index=pd.MultiIndex.from_tuples([('S_A', 3), ('S_A', 6), ('S_B', 3), ('S_C', 9)],
                                            names=['sid', 'k']))
        data.numbered_frame = pd.DataFrame(
            {'v': [1.0, 2.0, 3.0]},
            index=pd.MultiIndex.from_tuples([(1, 10), (1, 20), (3, 10)], names=['a', 'b']))
        data.new_cells('table', 'lambda: pdn.table(src_frame)')
        data.new_cells('numbered', 'lambda: pdn.table(numbered_frame)')
        projection = model.new_space('Projection', refs={'data': data})
        projection.new_cells('sid_of', 'lambda p: ("S_A", "S_B", "S_X", "S_C")[p]')
        projection.new_cells('probe', PROBE)
        projection.new_cells('lowered', LOWERED)
        model.export(root / 'exported')
        live = [[repr(projection.probe(p)), projection.lowered(p)] for p in range(4)]
        prefixes = [projection.probe(p)[0] for p in range(4)]
    finally:
        model.close()
    return live, prefixes


def test_native_table_index_answers_as_the_helper(tmp_path):
    live, prefixes = _model(tmp_path)
    assert prefixes == [True, True, False, True]        # S_A, S_B, S_C are level-0 labels, S_X is not
    assert [row[1] for row in live] == [True, False, True, False]
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'for p in range(4):\n    m.Projection.probe(p); m.Projection.lowered(p)\n', encoding='utf-8')
    for package in ('exported', 'exported_cy'):
        if package == 'exported_cy':
            done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                                   '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
            assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
        out = subprocess.run([sys.executable, '-c', RUN.format(package=package)], cwd=tmp_path,
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
        lines = out.stdout.strip().splitlines()
        assert json.loads(lines[-2]) == live, (package, lines[-2], live)
        if package == 'exported_cy':
            assert lines[-1] == 'NativeTable'
            classes = (tmp_path / 'exported_cy/_m_Projection/_mx_classes.py')
            text = classes.read_text(encoding='utf-8') if classes.exists() else '\n'.join(
                p.read_text(encoding='utf-8') for p in (tmp_path / 'exported_cy').rglob('_mx_classes.py'))
            assert '.has_1(' in text            # the lowered prefix test (D27a)
            assert 'sid in index' in text       # the local stays a Python read
