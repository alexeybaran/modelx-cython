"""A window proof that must survive.

A cell shaped like a segment balance (local ``m = t - age``, recursive call ``f(t - 1, t - 1 - m)``, argument
``t - m``) keeps its window proof: the cache entry carries ``_mx_proven_axes``, ``m`` is a plain C
difference, and no guard is emitted, so a later change of pass order cannot silently undo it.
"""

import subprocess
import sys
import textwrap

import modelx as mx

def test_seg_shaped_cell_keeps_its_window_proof(tmp_path):
    model = mx.new_model('D82Seg')
    try:
        values = model.new_space('Values')
        values.new_cells('seg_share', textwrap.dedent('''\
            def seg_share(t, segment_age):
                m = t - segment_age
                return 0.5 + 0.01 * m
            '''))
        values.new_cells('seg_bal', textwrap.dedent('''\
            def seg_bal(t, segment_age):
                m = t - segment_age
                if t == 0 or segment_age == 0:
                    return 1.0
                return seg_bal(t - 1, t - 1 - m) * seg_share(t, t - m)
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'v = mx_model.Values\n'
        'for t in range(24):\n'
        '    for s in range(6):\n'
        '        v.seg_bal(t, s)\n'
        '        v.seg_share(t, s)\n', encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({
        'dense_bounds_proofs': True, 'dense_one_axis': True, 'dense_miss_binding': True,
        'propagate_exact_receiver': True}), encoding='utf-8')
    subprocess.run(
        [sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--spec', 'spec.py'],
        cwd=tmp_path, check=True)
    probe = textwrap.dedent('''\
        from exported_cy import mx_model
        v = mx_model.Values
        def seg_bal(t, s):
            m = t - s
            if t == 0 or s == 0:
                return 1.0
            return seg_bal(t - 1, t - 1 - m) * (0.5 + 0.01 * (t - (t - m)))
        for t in range(24):
            for s in range(6):
                assert v.seg_bal(t, s) == seg_bal(t, s), (t, s)
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
