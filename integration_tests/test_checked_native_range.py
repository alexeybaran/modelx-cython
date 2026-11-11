"""Native unit-step range counters retain Python's bounds and binding semantics."""

import subprocess
import sys
import textwrap

import modelx as mx


def test_checked_native_range(tmp_path):
    model = mx.new_model('CheckedNativeRange')
    try:
        values = model.new_space('Values')
        values.new_cells('term', 'lambda index: float(index)')
        values.new_cells('step', 'lambda: 3')
        values.new_cells('add', textwrap.dedent('''\
            def add(origin, stop):
                total: float = 0.0
                for index in range(origin + 1, stop):
                    total += term(index) * ((index - origin) / 12.0)
                return total
            '''))
        values.new_cells('add_from_one', textwrap.dedent('''\
            def add_from_one(stop):
                total: float = 0.0
                for slot in range(1, stop):
                    total += term(slot)
                return total
            '''))
        values.new_cells('weighted', textwrap.dedent('''\
            def weighted(origin, stop, limit):
                step_size = step()
                total: float = 0.0
                for number in range(1, stop):
                    month = origin + (number - 1) * step_size
                    if number * step_size <= limit:
                        total += term(month) * ((month - origin) / 12.0)
                return total
            '''))
        values.new_cells('square', textwrap.dedent('''\
            def square(origin, stop):
                total = 0.0
                for squared_index in range(origin + 1, stop):
                    total += float(squared_index * squared_index)
                return total
            '''))
        values.new_cells('shadowed', textwrap.dedent('''\
            def shadowed(origin, stop):
                range = lambda a, b: (a,)
                total = 0.0
                for shadowed_index in range(origin + 1, stop):
                    total += term(shadowed_index)
                return total
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'v = mx_model.Values\n'
        'v.add(0, 5); v.add_from_one(5); v.weighted(0, 5, 5.5); v.square(0, 5); v.shadowed(0, 5)\n',
        encoding='utf-8')
    subprocess.run(
        [sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
        cwd=tmp_path, check=True)
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert 'index: _mx_cy.longlong\n' in source
    assert 'slot: _mx_cy.longlong\n' in source
    assert 'number: _mx_cy.longlong\n' in source and 'month: _mx_cy.longlong =' in source
    assert 'squared_index: _mx_cy.longlong' not in source
    assert 'shadowed_index: _mx_cy.longlong' not in source
    assert '9223372036854775806' in source
    probe = textwrap.dedent('''\
        from exported_cy import mx_model
        v = mx_model.Values
        bound = (1 << 63) - 1
        for origin, stop in ((-7, 8), (4, 4), (-bound - 1, -bound + 2), (bound, bound)):
            expected = 0.0
            for index in range(origin + 1, stop):
                expected += float(index) * ((index - origin) / 12.0)
            assert v.add(origin, stop) == expected
        assert v.square(bound - 2, bound) == float((bound - 1) ** 2)
        assert v.add_from_one(-bound) == 0.0 and v.add_from_one(5) == 10.0
        for origin in (0, -(1 << 63), (1 << 63) - 16):
            expected = sum(float(origin + (k - 1) * 3) * ((k - 1) * 3 / 12.0)
                           for k in range(1, 5) if k * 3 <= 11.5)
            assert v.weighted(origin, 5, 11.5) == expected
        assert v.shadowed(5, 5) == 6.0
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
