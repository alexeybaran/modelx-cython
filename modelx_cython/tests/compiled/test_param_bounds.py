"""Declared cache bounds on Direct and Collect calls, size inheritance, and enum domains."""
import json
import os
import subprocess
import sys
import textwrap
from http import HTTPStatus

import pytest
import modelx as mx


def _build(tmp_path, spec):
    model = mx.new_model('ParamBounds')
    try:
        keys = (1, 2, 3)
        p = model.new_space('Projection', formula='lambda policy_id: None',
                            refs={'keys': keys, 'Status': HTTPStatus})
        p.set_collect(over={'policy_id': 'keys'}, outputs=lambda: {"sum": {"total": total()}})
        p.new_cells('proj_len', 'lambda: 4 + 3 * policy_id if policy_id < 9 else 1 // 0')
        p.new_cells('cf', 'lambda t: 1.0 * t + policy_id')
        p.new_cells('pair', 'lambda t, k: cf(t) + k')
        p.new_cells('total', 'lambda: sum(cf(t) for t in range(proj_len())) + pair(0, 0)')
        p.new_cells('rate', textwrap.dedent('''\
            def rate(s):
                return 0.5 * int(s)
            '''))
        p.new_cells('status', 'lambda: Status.OK')
        p.new_cells('rated', 'lambda: rate(status())')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(textwrap.dedent('''\
        from exported import mx_model as m
        m.Projection[1].total()
        m.Projection[1].rated()
        '''))
    (tmp_path / 'spec.py').write_text(repr(spec))
    build = subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported',
                            '--sample', 'sample.py', '--spec', 'spec.py'],
                           cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True)
    (tmp_path / 'build.log').write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    return json.loads((tmp_path / 'exported_cy' / '_mx_admission.json').read_text())


def _cell(report, name):
    return next(v for k, v in report.items() if k.endswith('._f_' + name))


def _domain(entry):
    return entry['dense_logical_domain'] or [entry['associative_logical_domain'][k]
                                             for k in entry['arguments']]


@pytest.mark.parametrize('layer', ['model', 'space'])
def test_param_bounds_inherit_and_collect_check(tmp_path, layer):
    spec = {'parallel_workers': 1, 'cells_param_size': {'t': 11}}
    if layer == 'space':
        # the Space overrides the model for t; an exact cells_param_size entry wins for its axes
        spec['spaces'] = {'Projection': {'cells_param_size': {'t': 20, ('t', 'k'): (14, 1)}}}
    report = _build(tmp_path, spec)
    cf, pair = _cell(report, 'cf'), _cell(report, 'pair')
    stop = 11 if layer == 'model' else 20
    assert _domain(cf) == [[0, stop]]
    assert _domain(pair) == ([[0, 11], [0, 1]] if layer == 'model' else [[0, 14], [0, 1]])
    assert cf['declared_sizes'] == {'t': 'model' if layer == 'model' else 'Projection'}
    # a declared size is allocated as declared (no traced window with a spill path)
    assert cf['dense_window'] == [[0, stop]] and not cf['exact_spill']
    probe = textwrap.dedent('''\
        import sys
        import pytest
        from exported import mx_model as exported
        from exported_cy import mx_model as compiled

        stop = int(sys.argv[1])
        live, native = exported.Projection, compiled.Projection
        compiled._mx_collect_workers = 1

        for t in (0, stop - 1):
            assert native[2].cf(t) == live[2].cf(t) == t + 2.0
        assert native[2].total() == live[2].total() == 67.0
        for t in (-1, stop):
            with pytest.raises(IndexError) as error:
                native[3].cf(t)
            assert 'Projection.cf' in str(error.value) and f't in [0, {stop})' in str(error.value)

        live.keys = native.keys = (1, 2)
        assert native.collect().sum.total == live.collect().sum.total == 96.0

        live.keys = native.keys = (1, 2, 3)
        assert live.collect().sum.total == 216.0
        if stop < 13:
            # A rejected coordinate must fail again, even after earlier items finished.
            for _ in range(2):
                with pytest.raises(IndexError) as error:
                    native.collect()
                assert 'Projection.cf' in str(error.value) and f't in [0, {stop})' in str(error.value)
        else:
            assert native.collect().sum.total == 216.0

        # The formula itself raises before attempting an out-of-range Cell call.
        live.keys = native.keys = (1, 9)
        for projection in (live, native):
            with pytest.raises(ZeroDivisionError):
                projection.collect()

        # A different domain order forces a fresh evaluation after failure.
        live.keys = native.keys = (2, 1)
        assert native.collect().sum.total == live.collect().sum.total == 96.0
        ''')
    result = subprocess.run([sys.executable, '-B', '-c', probe, str(stop)], cwd=tmp_path,
                            env=dict(os.environ), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_enum_argument_domain_is_the_enum(tmp_path):
    report = _build(tmp_path, {'parallel_workers': 1,
                               'spaces': {'Projection': {'serial_collect': 'reuse'}}})
    rate = _cell(report, 'rate')
    top = max(int(member) for member in HTTPStatus)
    assert _domain(rate) == [[0, top + 1]], rate
    assert 'HTTPStatus' in rate['enum_domains']['s']
    probe = textwrap.dedent('''\
        from http import HTTPStatus
        from exported_cy import mx_model as m
        p = m.Projection[2]
        # the sample passed OK only; an unused member works as it does live
        for member in (HTTPStatus.OK, HTTPStatus.NOT_FOUND, max(HTTPStatus)):
            assert p.rate(member) == 0.5 * int(member)
        ''')
    result = subprocess.run([sys.executable, '-B', '-c', probe], cwd=tmp_path,
                            env=dict(os.environ), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
