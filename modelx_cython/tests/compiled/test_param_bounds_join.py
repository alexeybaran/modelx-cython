"""Enum domains of an IntEnum parameter on a model compiled with collect (the build of
``test_param_bounds_collect``). The class is an enum family (values 0..n-1), so the parameter is held as the
member's position over the whole class (enums.plan); a class that is no family (``HTTPStatus``) is an object."""
import os
import subprocess
import sys
import textwrap

from test_param_bounds_collect import _build, _cell, _domain


PHASES = 'import enum\n\nclass Phase(enum.IntEnum):\n    OK = 0\n    WAIT = 1\n    PAY = 2\n    END = 3\n'


def test_enum_argument_domain_is_the_enum(tmp_path, monkeypatch):
    (tmp_path / 'bounds_phase.py').write_text(PHASES, encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    from bounds_phase import Phase
    report = _build(tmp_path, {'parallel_workers': 1,
                               'spaces': {'Projection': {'serial_collect': 'reuse'}}}, status=Phase)
    rate = _cell(report, 'rate')
    assert _domain(rate) == [[0, len(Phase)]], rate
    assert 'Phase' in rate['enum_domains']['s']
    probe = textwrap.dedent('''\
        from bounds_phase import Phase
        from exported_cy import mx_model as m
        p = m.Projection[2]
        # the sample passed OK only; an unused member works as it does live
        for member in (Phase.OK, Phase.PAY, max(Phase)):
            assert p.rate(member) == 0.5 * int(member)
        ''')
    result = subprocess.run([sys.executable, '-B', '-c', probe], cwd=tmp_path,
                            env=dict(os.environ), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
