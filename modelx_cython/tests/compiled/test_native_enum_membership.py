"""``x in (E.A, E.B)`` / ``not in`` over enum members of a value that is not held as a position (an integer, an
object, members of a class that is no family): the members stay, and the membership gives what live gives."""
from http import HTTPStatus
import subprocess
import sys
import textwrap

import modelx as mx


def test_member_tuple_membership(tmp_path):
    model = mx.new_model('EnumMembership')
    try:
        s = model.new_space('Projection', refs={'Status': HTTPStatus})
        s.new_cells('kind', 'lambda t: 200 + t')
        s.new_cells('hit', 'lambda t: kind(t) in (Status.OK, Status.CREATED)')
        s.new_cells('miss', 'lambda t: kind(t) not in (Status.OK, Status.ACCEPTED, Status.CREATED)')
        s.new_cells('single', 'lambda t: kind(t) in (Status.OK,)')
        s.new_cells('single_not', 'lambda t: kind(t) not in (Status.OK,)')
        s.new_cells('local', textwrap.dedent('''\
            def local(t):
                value = kind(t)
                if value in (Status.ACCEPTED, Status.OK):
                    return 1.5
                return 0.5
            '''))
        s.new_cells('reversed_eq', 'lambda t: Status.CREATED == kind(t)')
        s.new_cells('argument', 'lambda k: k in (Status.OK, Status.NO_CONTENT)')
        s.new_cells('loose', 'lambda x: x in (Status.OK, Status.CREATED)')
        s.new_cells('literal_mix', 'lambda t: kind(t) in (Status.OK, 201)')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'p = m.Projection\n'
        'for t in range(-3, 9):\n'
        '    p.hit(t); p.miss(t); p.single(t); p.single_not(t); p.local(t); p.reversed_eq(t)\n'
        '    p.literal_mix(t)\n'
        'for k in (200, 204, 205, -1): p.argument(k)\n'
        'for x in (200, "a", 201.0, 202, None): p.loose(x)\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    generated = (tmp_path / 'exported_cy/_mx_classes.py').read_text()
    assert 'x in (self.Status.OK, self.Status.CREATED)' in generated, generated
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        p = import_module(sys.argv[1]).mx_model.Projection
        for t in range(-3, 9):
            k = 200 + t
            assert p.hit(t) is (k in (200, 201)), t
            assert p.miss(t) is (k not in (200, 202, 201)), t
            assert p.single(t) is (k == 200) and p.single_not(t) is (k != 200)
            assert p.local(t) == (1.5 if k in (202, 200) else 0.5)
            assert p.reversed_eq(t) is (k == 201)
            assert p.literal_mix(t) is (k in (200, 201))
        for k in (200, 204, 205, -1):
            assert p.argument(k) is (k in (200, 204))
        for x in (200, 'a', 201.0, 202, None):
            assert p.loose(x) is (x in (200, 201))
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True)
