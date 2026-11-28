"""Use all members of the traced IntEnum's domain."""
import json
import subprocess
import sys
import textwrap
from http import HTTPStatus

import modelx as mx


def _build(tmp_path):
    model = mx.new_model('ParamBounds')
    try:
        p = model.new_space('Projection', formula='lambda policy_id: None',
                            refs={'Status': HTTPStatus})
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
        m.Projection[1].rated()
        '''), encoding='utf-8')
    build = subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported',
                            '--sample', 'sample.py', '--no-spec'],
                           cwd=tmp_path, capture_output=True, text=True)
    (tmp_path / 'build.log').write_text(build.stdout + build.stderr, encoding='utf-8')
    assert build.returncode == 0, build.stdout + build.stderr
    return json.loads((tmp_path / 'exported_cy' / '_mx_admission.json').read_text(encoding='utf-8'))


def _cell(report, name):
    return next(v for k, v in report.items() if k.endswith('._f_' + name))


def _domain(entry):
    return entry['dense_logical_domain'] or [entry['associative_logical_domain'][k]
                                             for k in entry['arguments']]


def test_enum_argument_domain_is_the_enum(tmp_path):
    report = _build(tmp_path)
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
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
