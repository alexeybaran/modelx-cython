"""Enum domains of an IntEnum parameter on the direct route: the dense domain of a parameter traced as one
IntEnum class is the whole enum. Direct-route twin of ``test_param_bounds_join`` (the build of
``test_param_bounds``, without ``collect_over``)."""
import os
import subprocess
import sys
import textwrap
from http import HTTPStatus

from test_param_bounds import _build, _cell, _domain


def test_enum_argument_domain_is_the_enum(tmp_path):
    report = _build(tmp_path, {})
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
