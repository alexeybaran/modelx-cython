"""S1: dense cache values are not zero-filled when an item is allocated (``value_storage.py``, ``_mx_sys.pxd``).

Each model is built three times: as shipped; with ``MX_POISON_UNWRITTEN_VALUES``, which fills every value array of a
new object with 0xA5 bytes (the plain model also with heap types, ``CYTHON_USE_TYPE_SPECS``); and with
``MX_TEST_VALUES_FALLBACK``, which makes the layout check fail (plain model: a RuntimeWarning names each class) or the
mechanism unavailable (collect model: silent), so every byte is zeroed as before. Every build must give the exported
model's values on the direct route (new items, and the same keys again after release), on collect (a reused root,
one and four workers, a shared dense cache filled concurrently), on a Python subclass (zero-filled as before, and
copied through the class's own reduce), and after a copy or pickle round trip, where cached values survive. The
reduced state of a fresh and of a partly filled item is the same in all three builds, so no unwritten byte is carried;
the reduce has Cython's name; a reduce leaves no block and no type reference behind. On the free-threaded interpreter
one thread fills an item's caches and assigns its public attributes while another copies (and, without collect,
pickles) it: no crash, and every restored item returns the live model's values. The same holds for a locked Space,
whose flags are ``bint``.
"""
import os
import subprocess
import sys
import sysconfig
import textwrap

import modelx as mx


PROBE = textwrap.dedent("""\
    import copy, gc, pickle, sys
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
        clones = [('copy', copy.copy)]
        if not collect:                      # a model with collect holds a thread-local state: no pickle
            clones += [('pickle', lambda x: pickle.loads(pickle.dumps(x))), ('deepcopy', copy.deepcopy)]
        # the Python subclass is copied through the class's own reduce (zero-filled, not scrubbed)
        out['subclass_clones'] = [(type(q) is Derived, q.factor(3), q.factor(40))
                                  for q in (clone(d) for name, clone in clones)]
        p = m.Policy[5]
        p.pv(); p.counted(4); p.counted(6)
        seen = p.seen[0]
        for name, clone in clones:
            q = clone(p)
            # cached values survive: the counter of the formula does not move
            out[name] = (q.pv() == p.pv(), q.grid(7, 2) == p.grid(7, 2), q.n(11), q.alive(3),
                         q.counted(4), q.counted(6), q.seen[0] - seen)
        T = type(p)
        def drift(reduce, calls=1000):       # a scrubbed copy kept per call would be 1000 blocks
            for _ in range(100):
                reduce()
            gc.collect()
            before = sys.getallocatedblocks()
            for _ in range(calls):
                reduce()
            gc.collect()                     # also merges per-thread type references (free-threaded)
            return sys.getallocatedblocks() - before
        gc.collect()
        refs = sys.getrefcount(T)
        out['blocks_per_1000_reduces'] = min(drift(p.__reduce__) for _ in range(3))
        gc.collect()
        out['type_references_left'] = sys.getrefcount(T) - refs
        out['installed'] = sys.modules[T.__module__]._mx_uninitialised_values
        out['heap_types'] = bool(T.__flags__ & (1 << 9))
        out['reduce'] = [(type(R).__name__, R.__name__, R.__qualname__, R.__doc__)
                         for R in (T.__reduce__, type(m.Data).__reduce__)]
        def plain(value):                    # the reduced state, objects by type name; bool stays bool
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value if value is None or type(value) in (bool, int, float, str) else type(value).__name__
        fresh = m.Policy[7]
        fresh.grid(1, 1)
        out['states'] = [plain(x.__reduce__()) for x in (m.Policy[8], fresh, p, m.Data)]
    print(repr(out))
    """)

THREADS = textwrap.dedent("""\
    import copy, pickle, random, sys, threading
    native = __import__(sys.argv[1]).mx_model
    live = __import__(sys.argv[2]).mx_model
    mode = sys.argv[3]                       # plain: Policy items; collect: the shared Data; locked: Market
    if getattr(sys, '_is_gil_enabled', lambda: True)():   # before 3.13 there is always a GIL
        print(repr({'gil': True}))
        sys.exit(0)
    months = range(48)

    def entries(item):                       # every entry of the item's dense caches
        if mode == 'collect':
            return [(item.factor, (t,)) for t in months]
        if mode == 'locked':                 # bint flags, stored with release semantics under the model lock
            return [(item.price, (t,)) for t in months] + [(item.up, (t,)) for t in months]
        return ([(item.cf, (t,)) for t in months] + [(item.n, (t,)) for t in months]
                + [(item.alive, (t,)) for t in months] + [(item.counted, (t,)) for t in months]
                + [(item.grid, (t, j)) for t in months for j in range(3)])

    def values(item):                        # a missing entry is computed, a cached one returned
        return [cell(*args) for cell, args in entries(item)] + ([item.pv()] if mode == 'plain' else [])

    def item_of(model, key):
        return model.Policy[key] if mode == 'plain' else model.Data if mode == 'collect' else model.Market

    keys = list(range(1, 41)) if mode == 'plain' else [None]
    expected = {key: values(item_of(live, key)) for key in keys}
    # a model with collect holds a thread-local state, a locked model a lock: neither pickles
    clones = [copy.copy] + ([lambda x: pickle.loads(pickle.dumps(x))] if mode == 'plain' else [])
    current, copies, mismatches, errors = [None], [0], [], []
    done = threading.Event()

    def copier():                            # copies (and pickles) the item being filled, in a loop
        try:
            while current[0] is None and not done.is_set():
                pass
            while not done.is_set():
                key, item = current[0]
                q = clones[copies[0] % len(clones)](item)
                got = values(q)
                if got != expected[key]:
                    mismatches.append((key, [(a, b) for a, b in zip(got, expected[key]) if a != b][:3]))
                copies[0] += 1
        except BaseException as error:
            errors.append(repr(error))
            done.set()

    def churn(item):                         # fresh but equal public members: the old ones are freed
        if mode == 'collect':
            item._name = ''.join(['Da', 'ta'])
        elif mode == 'locked':
            item.months = tuple(range(48))
        else:
            item.months = tuple(range(48))
            item.keys = tuple(range(1, 41))
            item.seen = [0]

    def filler(rounds, chunk):               # fills each new item in a random order, a chunk per copy
        try:
            order = random.Random(0)
            for r in range(rounds):
                model = type(native)()       # a new model per round: nothing is added to a model being pickled
                key = keys[r % len(keys)]
                item = item_of(model, key)
                todo = entries(item)
                order.shuffle(todo)
                current[0] = (key, item)
                for start in range(0, len(todo), chunk):
                    for cell, args in todo[start:start + chunk]:
                        churn(item)
                        cell(*args)
                    seen = copies[0]
                    while copies[0] < seen + 1 and not done.is_set():
                        churn(item)
                if done.is_set():
                    break
        except BaseException as error:
            errors.append(repr(error))
        finally:
            done.set()

    threads = [threading.Thread(target=copier), threading.Thread(target=filler, args=(40, 8 if mode == 'plain' else 2))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(repr({'gil': False, 'copies': copies[0], 'mismatches': mismatches[:5], 'errors': errors}))
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
                            '--spec', f'spec_{name}.py'], cwd=tmp_path, capture_output=True, text=True, env=env,
                           timeout=900)
    assert build.returncode == 0, build.stdout + build.stderr
    return build.stdout + build.stderr


def _run(tmp_path, script, *args):
    run = subprocess.run([sys.executable, '-c', script, *args], cwd=tmp_path, capture_output=True, text=True,
                         timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    return eval(run.stdout), run.stderr


# setuptools passes CPPFLAGS to unix-style compilers only; MSVC reads extra options from CL instead (it takes
# -D as /D), and does not echo them in the build log. The macros' effect is asserted below either way: the
# fallback build's warning and installed flags, the poisoned plain build's heap types.
MSVC = os.name == 'nt'


def _with_macros(env, flags):
    return dict(env, CPPFLAGS=flags, **({'CL': flags} if MSVC else {}))


def _check(tmp_path, mode):
    plain = dict(os.environ)
    plain.pop('CPPFLAGS', None)
    plain.pop('CL', None)
    poison = '-DMX_POISON_UNWRITTEN_VALUES'
    flags = {'shipped': '',
             'poisoned': poison + (' -DCYTHON_USE_TYPE_SPECS=1' if mode == 'plain' else ''),
             'fallback': f'{poison} -DMX_TEST_VALUES_FALLBACK={1 if mode == "plain" else 2}'}
    for build, cppflags in flags.items():
        env = _with_macros(plain, cppflags) if cppflags else plain
        log = _build(tmp_path, f'{build}_{mode}', mode == 'collect', env)
        assert MSVC or all(flag in log for flag in cppflags.split()), build
    source = (tmp_path / f'shipped_{mode}_cy/_mx_classes.pxd').read_text(encoding='utf-8')
    policy = source.split('cdef class _c_Policy')[1].split('cpdef')[0]
    members = [line.strip() for line in policy.splitlines() if line.strip().startswith('cdef ')]
    values = [line for line in members if ' _v_' in line and '[' in line.split(' _v_')[0]]
    assert len(values) == 5 and members[-5:] == values         # the value arrays are the last members
    assert values[-1] == 'cdef bint[48] _v_alive'               # widest elements first: no padding
    expected, _ = _run(tmp_path, PROBE, f'shipped_{mode}', mode)
    shared = {}
    for build in flags:
        package = f'{build}_{mode}_cy'
        got, stderr = _run(tmp_path, PROBE, package, mode)
        installed = build != 'fallback'
        assert got.pop('installed') == {'_c_Data': installed, '_c_Policy': installed}, package
        warned = [f'dense cache values of {package}._mx_classes.{cls} are zero-filled' in stderr
                  for cls in ('_c_Data', '_c_Policy')]
        assert warned == [build == 'fallback' and mode == 'plain'] * 2, stderr      # a failed check is reported
        assert got.pop('heap_types') is (build == 'poisoned' and mode == 'plain')
        assert abs(got.pop('blocks_per_1000_reduces')) <= 2, package
        assert got.pop('type_references_left') == 0, package
        reduce = got.pop('reduce')
        assert [name for kind, name, qualname, doc in reduce] == ['__reduce_cython__'] * 2
        shared.setdefault('reduce', reduce)
        shared.setdefault('states', got['states'])
        # the same descriptor and the same reduced states as a fully zeroed build: no unwritten byte is carried
        assert (reduce, got.pop('states')) == (shared['reduce'], shared['states']), package
        names = ('copy',) + (('pickle', 'deepcopy') if mode != 'collect' else ())
        for name in names:
            assert got.pop(name) == (True, True, 55, True, 8.0, 12.0, 0), name
        assert got.pop('subclass_clones') == [(True, *got['subclass'])] * len(names), package
        if mode == 'collect':
            serial = expected[('collect', 1)]
            assert got.pop(('collect', 1)) == serial
            assert abs(got.pop(('collect', 4)) - serial) <= 1e-12 * abs(serial)
        assert got == {key: value for key, value in expected.items() if key[0] != 'collect'}, package
        if build != 'fallback':
            _assert_threads(tmp_path, package, f'shipped_{mode}', mode)


def _assert_threads(tmp_path, package, live, mode):
    threads, stderr = _run(tmp_path, THREADS, package, live, mode)
    if sysconfig.get_config_var('Py_GIL_DISABLED'):
        assert threads['gil'] is False, stderr
        assert threads['errors'] == [] and threads['mismatches'] == [], (package, threads)
        assert threads['copies'] >= 50, (package, threads)


def test_unwritten_values_of_a_locked_space(tmp_path):
    """A locked Space has bint flags, published under the model lock: its values are left unwritten too, and an
    item copied while another thread fills it holds only values behind flags it saw set."""
    model = mx.new_model('LockedValues')
    try:
        market = model.new_space('Market', refs={'months': tuple(range(48))})
        market.new_cells('level', 'lambda: 10.0')
        market.new_cells('price', 'lambda t: level() + 0.5 * t')
        market.new_cells('up', 'lambda t: t % 3 == 0')
        model.export(tmp_path / 'locked', locked_spaces=['Market'])
    finally:
        model.close()
    (tmp_path / 'spec.py').write_text(repr({'spaces': {'Market': {'cells_param_size': {'t': 48}}}}), encoding='utf-8')
    (tmp_path / 'sample.py').write_text('from locked import mx_model as m\nm.Market.price(1); m.Market.up(2)\n',
                                        encoding='utf-8')
    env = _with_macros(dict(os.environ), '-DMX_POISON_UNWRITTEN_VALUES')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'locked', '--sample', 'sample.py',
                            '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True, env=env, timeout=900)
    assert build.returncode == 0 and (MSVC or '-DMX_POISON_UNWRITTEN_VALUES' in build.stdout + build.stderr), \
        build.stdout + build.stderr
    source = (tmp_path / 'locked_cy/_mx_classes.pxd').read_text(encoding='utf-8')
    assert 'cdef bint[48] _has_price' in source and 'cdef bint[48] _has_up' in source
    probe = ('import sys; from locked_cy import mx_model as m\n'
             'print(repr(sys.modules[type(m.Market).__module__]._mx_uninitialised_values))\n')
    installed, _ = _run(tmp_path, probe)
    assert installed == {'_c_Market': True}
    _assert_threads(tmp_path, 'locked_cy', 'locked', 'locked')


def test_unwritten_values_on_collect_and_direct_routes(tmp_path):
    _check(tmp_path, 'collect')


def test_unwritten_values_after_pickle_and_copy(tmp_path):
    _check(tmp_path, 'plain')
