"""Enum specialization must read the current exported Reference on a miss.

An IntEnum class whose values are not 0..n-1 (``HTTPStatus``) is no enum family (enums.plan): like another
value with the same members (here a namespace) it stays an ordinary Reference, read on every evaluation and
rebindable, and its members stay members. A family's Reference is fixed: test_enum_by_type."""
from http import HTTPStatus
from types import SimpleNamespace
import subprocess
import sys
import textwrap
import pytest

import modelx as mx


@pytest.mark.parametrize('code_owned', [False, True])
def test_enum_reference_replacement(tmp_path, code_owned):
    model = mx.new_model('EnumContract')
    try:
        status = HTTPStatus if code_owned else SimpleNamespace(OK=HTTPStatus.OK)
        s = model.new_space('Projection', formula='lambda policy: None', refs={'Status': status})
        s.new_cells('kind', 'lambda t: 200 + t')
        s.new_cells('matches', 'lambda t: kind(t) == Status.OK')
        s.new_cells('uncached', 'lambda t: t == Status.OK').is_cached = False
        s.new_cells('local', textwrap.dedent('''\
            def local(t):
                value = kind(t)
                return value != Status.OK
            '''))
        model.export(tmp_path/'exported')
    finally:
        model.close()
    (tmp_path/'sample.py').write_text('from exported import mx_model as m\n'
        'for t in range(5): m.Projection.matches(t); m.Projection.local(t)\n'
        'for policy in (11,22): m.Projection[policy].matches(0)\n'
        'm.Projection.uncached(200)\n')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    generated = (tmp_path/'exported_cy/_mx_classes.py').read_text()
    assert 'Status.OK' in generated, generated
    probe = textwrap.dedent('''\
        import sys
        from types import SimpleNamespace
        from importlib import import_module
        from http import HTTPStatus
        s = import_module(sys.argv[1]).mx_model.Projection
        assert (s.Status is HTTPStatus) == (sys.argv[2] == 'True')
        initial = s.Status
        assert s.uncached(200) is True and s.uncached(2**40) is False
        first = s[11]
        assert first.Status is initial
        assert first.matches(0) is True
        del s[11]
        assert s[11] is not first and s[11].Status is initial
        assert s.matches(0) is True
        assert s.local(0) is False
        s.Status = SimpleNamespace(OK=201)
        assert s.matches(0) is True
        assert s.matches(1) is True
        assert s.local(1) is False
        error = ValueError('member')
        class Broken:
            @property
            def OK(self):
                raise error
        s.Status = Broken()
        try:
            s.matches(2)
        except ValueError as actual:
            assert actual is error
        else:
            raise AssertionError('member exception lost')
        s.Status = SimpleNamespace(OK=202)
        assert s.matches(2) is True
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package, str(code_owned)], cwd=tmp_path, check=True)
