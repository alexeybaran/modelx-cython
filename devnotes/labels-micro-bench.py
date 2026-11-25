"""Micro models of the enum rule (devnotes/labels-rule-w5.md): build each tree with the modelx-cython first on
PYTHONPATH, then time fresh processes, alternating two built trees.

    PYTHONPATH=<checkout> python labels-micro-bench.py build <work dir>
    python labels-micro-bench.py compare <work dir A> <work dir B> [runs]
micro 1: a three-way `==` chain and an `in` on an IntEnum parameter, ~1.3 million Cell evaluations.
micro 2: sum(claims(t, k, s) for k in (A, B, C)) over a literal tuple of members.
"""
import os, subprocess, sys, textwrap, pathlib, shutil, statistics
import modelx as mx

N = 325_000
ENUMS = 'import enum\nclass Kind(enum.IntEnum):\n    X = 0\n    Y = 1\n    Z = 2\nclass Sex(enum.IntEnum):\n    M = 0\n    F = 1\n'
MODELS = {
    'micro1': {
        'f': textwrap.dedent('''\
            def f(t, k):
                if k == Kind.X:
                    return 1.0 * t
                if k == Kind.Y:
                    return 2.0 * t
                if k in (Kind.Z,):
                    return 3.0 * t
                return 0.0
            '''),
        'g': 'lambda t: f(t, Kind.X) + f(t, Kind.Y) + f(t, Kind.Z)',
        'run': 'lambda n: sum(g(t) for t in range(n))',
    },
    'micro2': {
        'claims': 'lambda t, k, s: (1.0 if k == Kind.X else 2.0 if k == Kind.Y else 3.0) * t * (1.5 if s == Sex.M else 1.0)',
        'total': 'lambda t, s: sum(claims(t, k, s) for k in (Kind.X, Kind.Y, Kind.Z))',
        'run': 'lambda n: sum(total(t, Sex.M) + total(t, Sex.F) for t in range(n))',
    },
}


def build(work):
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    (work / 'bench_enums.py').write_text(ENUMS)
    sys.path.insert(0, str(work))
    import bench_enums as e
    for name, cells in MODELS.items():
        d = work / name; d.mkdir()
        model = mx.new_model(name)
        s = model.new_space('P', refs={'Kind': e.Kind, 'Sex': e.Sex})
        for c, formula in cells.items():
            s.new_cells(c, formula)
        model.export(d / 'exported')
        model.close()
        (d / 'sample.py').write_text('from exported import mx_model as m\nm.P.run(40)\n')
        (d / 'spec.py').write_text(f"{{'cells_param_size': {{'t': {N}, 'n': {N + 1}}}}}\n")
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, (str(d), str(work), os.environ.get('PYTHONPATH'))))}
        done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                               '--spec', 'spec.py'], cwd=d, capture_output=True, text=True, env=env)
        assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]


def time_one(work, name, runs):
    d = work / name
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, (str(d), str(work), os.environ.get('PYTHONPATH'))))}
    code = ('import time\nfrom exported_cy import mx_model as m\nfrom exported import mx_model as live\n'
            f't0 = time.perf_counter(); v = m.P.run({N}); t1 = time.perf_counter()\nprint(t1 - t0, v)\n')
    out = []
    for _ in range(runs):
        r = subprocess.run([sys.executable, '-c', code], cwd=d, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        out.append(r.stdout.split())
    times = sorted(float(o[0]) for o in out)
    return times, out[0][1]


def compare(a, b, runs):
    for name in MODELS:
        ta, tb = [], []
        for _ in range(runs):
            ta += time_one(a, name, 1)[0]
            tb += time_one(b, name, 1)[0]
        print(f'{name}: A min {min(ta):.4f} median {statistics.median(ta):.4f} | B min {min(tb):.4f} '
              f'median {statistics.median(tb):.4f} | B/A min {min(tb) / min(ta):.3f} '
              f'median {statistics.median(tb) / statistics.median(ta):.3f}')


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(pathlib.Path(sys.argv[2]).resolve())
    else:
        compare(*(pathlib.Path(x).resolve() for x in sys.argv[2:4]), int(sys.argv[4]) if len(sys.argv) > 4 else 25)
