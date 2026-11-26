"""An IntEnum parameter's domain is its class: every member, not only the members the sample passed.

A parameter whose every traced value is a member of one IntEnum class gets the class's member range as its
dense cache domain (the build report names the class), so a member the sample never passed is cached and
gives live's value. Direct route; the collect half is ``test_enum_parameter_domain_collect``.
"""
import json
import os
import subprocess
import sys
import textwrap
from http import HTTPStatus

import modelx as mx


def build(tmp_path, collect=False):
    model = mx.new_model('EnumDomain')
    try:
        keys = (1, 2, 3)
        p = model.new_space('Projection', formula='lambda policy_id: None', refs={'keys': keys, 'Status': HTTPStatus})
        p.new_cells('rate', textwrap.dedent('''\
            def rate(s):
                return 0.5 * int(s) + policy_id
            '''))
        p.new_cells('status', 'lambda: Status.OK')
        p.new_cells('rated', 'lambda: rate(status())')
        p.new_cells('total', 'lambda: rated() + rate(Status.CREATED)')
        if collect:
            p.set_collect(over={'policy_id': keys}, outputs='lambda: {"sum": {"total": total()}}')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    sample = ('from exported import mx_model as m\nm.Projection.collect()\n' if collect else
              'from exported import mx_model as m\nm.Projection[1].total()\n')
    (tmp_path / 'sample.py').write_text(sample, encoding='utf-8')
    done = subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    report = json.loads((tmp_path / 'exported_cy' / '_mx_admission.json').read_text())
    return next(v for k, v in report.items() if k.endswith('._f_rate'))


PROBE = textwrap.dedent('''\
    from http import HTTPStatus
    from exported_cy import mx_model as m
    for k in (1, 2, 3):
        p = m.Projection[k]
        # the sample passed OK and CREATED only; any other member is cached and equals live
        for member in (HTTPStatus.OK, HTTPStatus.NOT_FOUND, max(HTTPStatus), min(HTTPStatus)):
            assert p.rate(member) == 0.5 * int(member) + k, (k, member)
        assert p.total() == 0.5 * 200 + 0.5 * 201 + 2 * k
    print('ok')
    ''')


def test_enum_parameter_domain_is_the_class(tmp_path):
    rate = build(tmp_path)
    top = max(int(member) for member in HTTPStatus)
    assert rate['cache'] == 'dense' and rate['dense_logical_domain'] == [[0, top + 1]], rate
    assert 'http.HTTPStatus' in rate['enum_domains']['s'], rate
    done = subprocess.run([sys.executable, '-B', '-c', PROBE], cwd=tmp_path, env=dict(os.environ),
                          capture_output=True, text=True)
    assert done.returncode == 0 and 'ok' in done.stdout, done.stdout + done.stderr
