import subprocess
import sys
import textwrap

import libcst as cst
import modelx as mx
import pytest

from modelx_cython.vector_ref_reads import _ScalarTail, _UnsupportedTail


def test_starred_index_does_not_enter_the_integer_vector_tail():
    node = cst.parse_expression('knots[*i]')
    tail = _ScalarTail('knots', 'held', None, None, None, None)
    with pytest.raises(_UnsupportedTail):
        tail.expression(node)


def test_relative_integer_vector_scalar_reads(tmp_path):
    model = mx.new_model('RefVectorReads')
    try:
        data = model.new_space('Data')
        data.values = [1000, 1002, 1005]
        data.new_cells('knots', 'def knots():\n    return [v for v in values]\n')
        space = model.new_space('Projection', formula='lambda policy: None', refs={'data': data})
        space.new_cells('weight', textwrap.dedent('''\
            def weight(i):
                knots = data.knots()
                current = knots[i]
                if i == len(knots) - 1:
                    return current + 0.5
                following = knots[i + 1]
                return (following - current) / 2.0
            '''))
        space.new_cells('identity', 'def identity():\n    v = data.knots()\n    return float(v[0] is v[0])\n')
        space.new_cells('branch_rebind', textwrap.dedent('''\
            def branch_rebind(i):
                v = data.knots()
                x = v[0]
                if i > 0:
                    x = v[1]
                return x + 0.5
            '''))
        space.new_cells('fractional_index', textwrap.dedent('''\
            def fractional_index(i):
                if i == 0:
                    return 0.0
                v = data.knots()
                return v[-(i / 2)] + 0.5
            '''))
        space.new_cells('prebound', textwrap.dedent('''\
            def prebound(i):
                x = []
                v = data.knots()
                x = v[i]
                return x + 0.5
            '''))
        model.export(tmp_path / 'exported', locked_spaces=['Data', 'Projection'])
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\n'
                                        'm.Projection(1).weight(0)\nm.Projection(1).identity()\n'
                                        'm.Projection(1).branch_rebind(1)\n'
                                        'm.Projection(1).fractional_index(0)\n'
                                        'm.Projection(1).prebound(1)\n')
    (tmp_path / 'spec.py').write_text(repr({'spaces': {'Projection': {
        'cells_param_size': {'i': (-5, 5)}, 'cells': {
            'weight': {'native_vector_ref_reads': True},
            'identity': {'native_vector_ref_reads': True},
            'branch_rebind': {'native_vector_ref_reads': True},
            'fractional_index': {'native_vector_ref_reads': True},
            'prebound': {'native_vector_ref_reads': True},
    }}}}))
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported'], cwd=tmp_path, check=True)
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    formula = source.split('def _f_weight(', 1)[1].split('def ', 1)[0]
    assert 'ivec_small_for_reads' in formula and '._mx_cell_knots()' in formula
    for name in ('identity', 'branch_rebind', 'fractional_index', 'prebound'):
        body = source.split(f'def _f_{name}(', 1)[1].split('def ', 1)[0]
        assert 'ivec_small_for_reads' not in body
    probe = '''\
from exported_cy import mx_model as m
from exported import mx_model as live
for i in (-1, 0, 1, 2):
    assert m.Projection(1).weight(i) == live.Projection(1).weight(i)
m.Data.knots()[0] = -1
assert m.Data.knots()[0] == 1000
assert m.Projection(1).identity() == live.Projection(1).identity() == 1.0
assert m.Projection(1).branch_rebind(1) == live.Projection(1).branch_rebind(1) == 1002.5
assert m.Projection(1).prebound(1) == live.Projection(1).prebound(1) == 1002.5
for model in (live, m):
    try:
        model.Projection(1).fractional_index(1)
    except TypeError:
        pass
    else:
        raise AssertionError('accepted a fractional index')
    try:
        model.Projection(1).weight(4)
    except IndexError as error:
        assert str(error) == 'list index out of range'
    else:
        raise AssertionError('accepted an out-of-range index')

wide = type(m)()
wide.Data.values = [-(1 << 63), (1 << 63) - 1]
assert wide.Projection(1).weight(0) == ((1 << 64) - 1) / 2.0

class DifferentData(type(m.Data)):
    def knots(self):
        return [15, 29, 47]

calc = m.Projection(2)
calc.data = DifferentData(m)
assert calc.weight(0) == 7.0
'''
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
