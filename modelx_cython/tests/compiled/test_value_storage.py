"""S1: dense cache values are not zero-filled when an item is allocated (``value_storage.py``, ``_mx_sys.pxd``).

Each model is built as shipped and with ``MX_POISON_UNWRITTEN_VALUES``, which fills every value array of a new
object with 0xA5 bytes. Every build must give the exported model's values on the direct route (new items, and
the same keys again after release), on collect (a reused root, one and four workers, a shared dense cache
filled concurrently), on a Python subclass (zero-filled as before), and after a copy or pickle round trip,
where cached values survive and no unwritten byte is carried.
"""
import os
import subprocess
import sys
import textwrap

import modelx as mx


PROBE = textwrap.dedent("""\
    import copy, pickle, struct, sys
    m = __import__(sys.argv[1]).mx_model
    collect = sys.argv[2] == 'collect'
    out = {}
    if collect:
        for workers in (1, 4):
            m = type(m)()                    # a new model instance per run
            m.collect_workers = workers
            out['collect', workers] = m.Policy.collect().sum.pv
        m = type(m)()
    for rep in range(3):                     # new items, then the same keys again after release
        for key in (1, 2, 5, 9):
            p = m.Policy[key]
            out['direct', rep, key] = (p.pv(), p.grid(7, 2), p.n(11), p.alive(3))
            del p
            del m.Policy[key]
    class Derived(type(m.Data)):
        pass
    d = Derived(m)
    d._mx_copy_refs(m.Data, m)
    out['subclass'] = (d.factor(3), d.factor(40))
    if sys.argv[1].endswith('_cy'):
        p = m.Policy[5]
        p.pv(); p.counted(4); p.counted(6)
        seen = p.seen[0]
        clones = [('copy', copy.copy)]
        if not collect:                      # a model with collect holds a thread-local state: no pickle
            clones += [('pickle', lambda x: pickle.loads(pickle.dumps(x))), ('deepcopy', copy.deepcopy)]
        for name, clone in clones:
            q = clone(p)
            # cached values survive: the counter of the formula does not move
            out[name] = (q.pv() == p.pv(), q.grid(7, 2) == p.grid(7, 2), q.n(11), q.alive(3),
                         q.counted(4), q.counted(6), q.seen[0] - seen)
        out['installed'] = sys.modules[type(p).__module__]._mx_uninitialised_values
        poison = {struct.unpack('<d', b'\\xa5' * 8)[0], int.from_bytes(b'\\xa5' * 8, 'little', signed=True)}
        def leaves(value):
            if isinstance(value, (list, tuple)):
                for item in value:
                    yield from leaves(item)
            else:
                yield value
        fresh = m.Policy[7]
        fresh.grid(1, 1)
        states = [fresh.__reduce__(), p.__reduce__(), m.Data.__reduce__()]
        out['poisoned_state'] = any(type(v) in (int, float) and v in poison for v in leaves(states))
    print(repr(out))
    """)


def _build(tmp_path, name, collect, env):
    model = mx.new_model('ValueStorage')
    try:
        data = model.new_space('Data')
        data.new_cells('factor', 'lambda t: 1.0 + 0.01 * t')
        keys = tuple(range(1, 41))
        options = {'collect_over': {'key': keys}, 'collect_outputs': 'lambda: {"sum": {"pv": pv()}}'} if collect else {}
        policy = model.new_space('Policy', formula='lambda key: None',
                                 refs={'Data': data, 'keys': keys, 'months': tuple(range(48)), 'seen': [0]}, **options)
        policy.new_cells('cf', 'lambda t: key * 0.5 * Data.factor(t)')
        policy.new_cells('n', 'lambda t: t * key')
        policy.new_cells('alive', 'lambda t: t < key')
        policy.new_cells('grid', 'lambda t, j: cf(t) * (j + 1)')
        policy.new_cells('counted', 'def counted(t):\n    seen[0] += 1\n    return t * 2.0\n')
        policy.new_cells('pv', 'lambda: sum(cf(t) + grid(t, t % 3) + n(t) * alive(t) for t in months)')
        model.export(tmp_path / name)
    finally:
        model.close()
    spec = {'spaces': {'Policy': {'serial_collect': 'reuse', 'parallel_collect_reuse': True}}} if collect else {}
    (tmp_path / f'spec_{name}.py').write_text(repr(spec), encoding='utf-8')
    run = 'm.Policy.collect()\n' if collect else 'm.Policy[1].pv(); m.Policy[40].pv()\n'
    (tmp_path / f'sample_{name}.py').write_text(
        f'from {name} import mx_model as m\n{run}p = m.Policy[3]\n'
        'p.grid(0, 0); p.grid(47, 2); p.counted(0); p.counted(47); p.n(0); p.alive(0)\n', encoding='utf-8')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', name, '--sample', f'sample_{name}.py',
                            '--spec', f'spec_{name}.py'], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert build.returncode == 0, build.stdout + build.stderr
    return build.stdout + build.stderr


def _run(tmp_path, package, mode):
    run = subprocess.run([sys.executable, '-c', PROBE, package, mode], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    return eval(run.stdout)


def _check(tmp_path, mode):
    plain = dict(os.environ)
    plain.pop('CPPFLAGS', None)
    _build(tmp_path, f'shipped_{mode}', mode == 'collect', plain)
    log = _build(tmp_path, f'poisoned_{mode}', mode == 'collect', dict(plain, CPPFLAGS='-DMX_POISON_UNWRITTEN_VALUES'))
    assert '-DMX_POISON_UNWRITTEN_VALUES' in log
    source = (tmp_path / f'shipped_{mode}_cy/_mx_classes.pxd').read_text(encoding='utf-8')
    policy = source.split('cdef class _c_Policy')[1].split('cpdef')[0]
    members = [line.strip() for line in policy.splitlines() if line.strip().startswith('cdef ')]
    values = [line for line in members if ' _v_' in line and '[' in line.split(' _v_')[0]]
    assert len(values) == 5 and members[-5:] == values         # the value arrays are the last members
    assert values[-1] == 'cdef bint[48] _v_alive'               # widest elements first: no padding
    expected = _run(tmp_path, f'shipped_{mode}', mode)
    for package in (f'shipped_{mode}_cy', f'poisoned_{mode}_cy'):
        got = _run(tmp_path, package, mode)
        assert got.pop('installed') == {'_c_Data': True, '_c_Policy': True}
        assert got.pop('poisoned_state') is False
        for name in ('copy',) + (('pickle', 'deepcopy') if mode != 'collect' else ()):
            assert got.pop(name) == (True, True, 55, True, 8.0, 12.0, 0), name
        if mode == 'collect':
            serial = expected[('collect', 1)]
            assert got.pop(('collect', 1)) == serial
            assert abs(got.pop(('collect', 4)) - serial) <= 1e-12 * abs(serial)
        assert got == {key: value for key, value in expected.items() if key[0] != 'collect'}, package


def test_unwritten_values_on_collect_and_direct_routes(tmp_path):
    _check(tmp_path, 'collect')


def test_unwritten_values_after_pickle_and_copy(tmp_path):
    _check(tmp_path, 'plain')
