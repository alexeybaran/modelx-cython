"""Differential helper semantics, independent of actuarial formulas."""
from pathlib import Path
import subprocess
import sys
import textwrap

from modelx_cython import runtime as native_runtime


def test_native_helper_operations(tmp_path):
    source = Path(native_runtime.__file__).parent
    for suffix in ('.pyx', '.pxd'):
        (tmp_path / ('native' + suffix)).write_bytes(
            (source / ('_native_inputs' + suffix)).read_bytes())
    functions = {
        'lookup': ('table', 'key', 'column'),
        'previous': ('table', 'key', 'column'),
        'contains': ('table', 'key'), 'key_count': ('table',),
        'key_at': ('table', 'index'), 'first_key': ('table',),
        'last_key': ('table',), 'subtable': ('table', 'key'),
        'has_group': ('table', 'key'), 'sum_f64': ('values',), 'sum_helper': ('values',),
    }
    wrapper = ['# cython: freethreading_compatible=True', 'cimport native']
    for name, parameters in functions.items():
        args = ', '.join(('function', *parameters))
        if name in ('lookup', 'previous'):
            wrapper += [f'def {name}({args}, *default):',
                        f'    if default: return native.{name}({args}, default[0])',
                        f'    return native.{name}({args})']
        else:
            wrapper += [f'def {name}({args}):', f'    return native.{name}({args})']
    (tmp_path / 'probe.pyx').write_text('\n'.join(wrapper) + '\n')
    (tmp_path / 'setup.py').write_text(
        'from setuptools import setup\nfrom Cython.Build import cythonize\n'
        'setup(ext_modules=cythonize(["native.pyx", "probe.pyx"], '
        'compiler_directives={"freethreading_compatible": True}))\n')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                   cwd=tmp_path, check=True)
    script = textwrap.dedent('''\
        import sys
        from enum import IntEnum
        from concurrent.futures import ThreadPoolExecutor
        import pickle
        from modelx_helpers.immutable_inputs import Table, Record
        from modelx_helpers import typed_inputs as py
        before = getattr(sys, '_is_gil_enabled', lambda: True)()
        import probe
        assert getattr(sys, '_is_gil_enabled', lambda: True)() == before

        class Key(IntEnum):
            FIRST = -2
            LAST = 9

        table = Table((9, -2, 5), ((90.0, 'nine'), (-20.0, 'minus'), (50.0, 'five')),
                      ('value', 'label'))
        enums = Table((Key.LAST, Key.FIRST), ((Key.LAST,), (Key.FIRST,)), ('value',))
        empty = Table((), (), ('value',))
        huge = Table((2**90, 1), ((9.0,), (1.0,)), ('value',))
        unordered_float = Table((1.0, float('nan'), -2.0), ((1,), (2,), (3,)), ('value',))
        record = Record({'table': table})
        cases = [
            ('lookup', (table, -2, 'value')), ('lookup', (table, 9, 'label')),
            ('lookup', (table, 5, -1)), ('lookup', (table, 5, 'missing')),
            ('lookup', (table, 5, 3)), ('lookup', (table, 4, 'value')),
            ('lookup', (table, 4, 'value', None)), ('lookup', (table, [], 0)),
            ('lookup', (enums, -2, 'value')),
            ('previous', (table, 8, 'value')), ('previous', (table, -3, 0)),
            ('previous', (table, -3, 0, None)), ('previous', (table, 8.5, 0)),
            ('previous', (huge, 2**89, 0)), ('previous', (empty, 0, 0, None)),
            ('previous', (enums, 8, 0)), ('contains', (table, 5)),
            ('contains', (table, 4)), ('key_count', (table,)),
            ('key_at', (table, 0)), ('key_at', (enums, 0)),
            ('key_at', (table, -1)), ('key_at', (table, 3)),
            ('first_key', (table,)), ('last_key', (table,)),
            ('first_key', (empty,)), ('last_key', (empty,)),
            ('first_key', (unordered_float,)), ('last_key', (unordered_float,)),
            ('subtable', (record, 'table')), ('has_group', (record, 'table')),
            ('has_group', (record, 'missing')), ('subtable', (record, 'missing')),
            ('sum_f64', ([1e16, 1.0, -1e16],)),
            ('sum_helper', ([1e16, 1.0, -1e16],)),
        ]
        def outcome(function, args):
            try:
                value = function(*args)
                return ('ok', type(value), value)
            except Exception as error:
                return ('error', type(error), error.args)
        for name, args in cases:
            original, native = getattr(py, name), getattr(probe, name)
            assert outcome(original, args) == outcome(native, (original, *args)), (name, args)
            sentinel = object()
            replacement = lambda *args: sentinel
            assert native(replacement, *args) is sentinel
        assert probe.lookup(py.lookup, enums, -2, 0) is Key.FIRST
        assert probe.subtable(py.subtable, record, 'table') is table
        assert probe.key_at(py.key_at, enums, 0) is Key.LAST
        restored = pickle.loads(pickle.dumps(table))
        assert not restored._native_cache
        with ThreadPoolExecutor(4) as pool:
            values = list(pool.map(lambda key: probe.previous(py.previous, restored, key, 0),
                                   [8] * 100))
        assert values == [50.0] * 100
        import math
        from decimal import Decimal
        from fractions import Fraction
        class FloatSubclass(float):
            pass
        for items in ([], [1.0, 2.0], [1e16, 1.0, -1e16],
                      [1e16, 1, -1e16], [float('inf'), -float('inf')],
                      [1e308, 1e308], [float('nan')], [-0.0],
                      [2**100, 1.0], [2**2000], [Decimal('1.0')],
                      [Fraction(1, 3), Fraction(2, 3)],
                      [FloatSubclass(2.0), 1e16, 1.0, -1e16],
                      [complex(1, 2), complex(3, -2)]):
            def evaluate(function):
                try:
                    value = function(x for x in items)
                    return ('ok', type(value), repr(value))
                except Exception as error:
                    return ('error', type(error), error.args)
            assert evaluate(py.sum_helper) == evaluate(
                lambda values: probe.sum_helper(py.sum_helper, values)), items
        error = ValueError('iteration')
        def broken():
            yield 1.0
            raise error
        try:
            probe.sum_helper(py.sum_helper, broken())
        except ValueError as actual:
            assert actual is error
        else:
            raise AssertionError('iteration error lost')
        ''')
    subprocess.run([sys.executable, '-c', script], cwd=tmp_path, check=True)
