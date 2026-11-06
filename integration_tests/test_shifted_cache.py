"""Observed negative bounds size a cache without closing its key domain."""
import subprocess
import sys
import textwrap

import modelx as mx


def test_shifted_dense_and_exact_spill(tmp_path):
    model = mx.new_model('ShiftedCaches')
    try:
        for name in ('Plain', 'Locked'):
            s = model.new_space(name, refs={'seen': []})
            s.new_cells('value', textwrap.dedent('''\
                def value(t):
                    seen.append(t)
                    return t * 1.0
                '''))
            s.new_cells('legacy', 'lambda t: t * 1.0')
        model.export(tmp_path/'exported', locked_spaces=['Locked'])
    finally:
        model.close()
    (tmp_path/'sample.py').write_text('from exported import mx_model as m\n'
        'for s in (m.Plain, m.Locked):\n'
        '    for t in (-1, 0, 3): s.value(t)\n'
        '    s.legacy(0); s.legacy(3)\n')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    generated = (tmp_path/'exported_cy/_mx_classes.py').read_text()
    assert '_spill_value' in generated
    probe = textwrap.dedent('''\
        from exported_cy import mx_model as m
        for s in (m.Plain, m.Locked):
            for t in (-1, 0, 3, -5, 9):
                assert s.value(t) == s.value(t) == float(t)
            assert s.seen == [-1, 0, 3, -5, 9]
            try:
                s.legacy(-1)
            except IndexError:
                pass
            else:
                raise AssertionError('legacy IndexRange changed')
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
