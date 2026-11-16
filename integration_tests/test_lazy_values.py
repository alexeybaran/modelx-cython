"""The dense value arrays of a compiled class are not zeroed at allocation (lazy_values).

A value slot is read only after its readiness flag is set. These tests dirty the memory first (items
computed, released, and new items with other keys allocated in their place) and check that every cell
of the new items, the collect root after them, an item after clear_all and an instance of a Python
subclass (generic allocation) compute the exported model's values. A stress test runs eight threads that
construct, fill, release and re-create items and collect on their own model instances while another
thread runs gc.collect() in a loop.
"""
import json
import re
import subprocess
import sys
import textwrap

import modelx as mx

KEYS = tuple(range(1, 121))


def _build(tmp_path):
    model = mx.new_model('LazyValues')
    try:
        projection = model.new_space(
            'Projection', formula='lambda key: None', refs={'keys': KEYS},
            collect_over={'key': KEYS}, collect_outputs='lambda: {"sum": {"value": value()}}')
        projection.new_cells('base_rate', 'lambda t: 0.001 * (t + 1)')          # independent of the key
        projection.new_cells('rate', 'lambda t: base_rate(t) + 0.0001 * ((t * key) % 13)')
        projection.new_cells('surv', 'lambda t: 1.0 if t == 0 else surv(t - 1) * (1.0 - rate(t - 1))')
        projection.new_cells('count', 'lambda t: (t * key) % 7')
        projection.new_cells('flag', 'lambda t: (t + key) % 3 == 0')
        projection.new_cells('value', textwrap.dedent('''\
            def value():
                total = 0.0
                for t in range(40):
                    total += surv(t) * (count(t) + 1) + (1.0 if flag(t) else 0.0)
                return total
            '''))
        portfolio = model.new_space('Portfolio', refs={'projection': projection})
        portfolio.new_cells('direct', textwrap.dedent('''\
            def direct():
                total = 0.0
                for key in projection.keys:
                    item = projection[key]
                    total += item.value()
                return total
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Projection\n'
        'p.collect(); mx_model.Portfolio.direct()\n'
        '[p.base_rate(t) for t in range(40)]\n', encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                          cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return done.stdout + done.stderr


VALUES = textwrap.dedent('''\
    import gc, json, sys
    from importlib import import_module
    module = sys.argv[1]
    m = import_module(module).mx_model
    p = m.Projection
    keys = list(p.keys)
    first, second = keys[:60], keys[60:]
    def cells(item):
        return [item.value()] + [item.rate(t) for t in range(40)] + [item.surv(t) for t in range(40)] \\
            + [float(item.count(t)) for t in range(40)] + [float(item.flag(t)) for t in range(40)]
    out = {}
    if module == 'exported_cy':
        classes = import_module('exported_cy._mx_classes')
        out['installed'] = classes._mx_lazy_values_installed
    for k in first:                     # fill memory with other values, then release it
        cells(p[k])
    for k in first:
        del p[k]
    gc.collect()
    out['second'] = [cells(p[k]) for k in second]       # allocated into the released memory
    for k in second:
        del p[k]
    m.Portfolio.clear_all()
    out['collect'] = p.collect().sum.value              # the collect root, also on released memory
    out['direct'] = m.Portfolio.direct()
    item = p[keys[3]]
    cells(item)
    p.clear_all()
    out['after_clear'] = cells(p[keys[3]])
    if module == 'exported_cy':
        Sub = type('Sub', (type(p),), {})                 # a heap subclass: the generic allocation
        sub = Sub(p._parent)
        sub._mx_copy_refs(p, p)
        out['subclass'] = [sub.base_rate(t) for t in range(40)]
    else:
        out['subclass'] = [p.base_rate(t) for t in range(40)]
    print(json.dumps(out))
    ''')

STRESS = textwrap.dedent('''\
    import gc, sys, threading
    from exported_cy import mx_model as template
    errors, stop = [], threading.Event()
    def collector():
        while not stop.is_set():
            gc.collect()
    def worker(index):
        try:
            m = type(template)()
            m._mx_collect_set_workers_v2(1)
            p = m.Projection
            keys = list(p.keys)
            expected = m.Portfolio.direct()
            for round_ in range(4):
                chosen = keys[index::8]
                for k in chosen:
                    p[k].value()
                for k in chosen:
                    del p[k]
                m.Portfolio.clear_all()
                if abs(m.Portfolio.direct() - expected) > 1e-7 + 1e-10 * abs(expected):
                    errors.append(f'direct {index} {round_}')
                p.clear_all()
                if abs(p.collect().sum.value - expected) > 1e-7 + 1e-10 * abs(expected):
                    errors.append(f'collect {index} {round_}')
        except Exception as error:  # noqa: BLE001
            errors.append(f'{index}: {type(error).__name__}: {error}')
    gc_thread = threading.Thread(target=collector)
    gc_thread.start()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    gc_thread.join()
    print('errors', errors)
    sys.exit(1 if errors else 0)
    ''')


def _run(tmp_path, code, *args):
    done = subprocess.run([sys.executable, '-c', code, *args], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def _close(a, b):
    return abs(a - b) <= 1e-7 + 1e-10 * abs(b)


def test_unzeroed_value_arrays_are_never_read_before_their_flag(tmp_path):
    _build(tmp_path)
    live = json.loads(_run(tmp_path, VALUES, 'exported'))
    compiled = json.loads(_run(tmp_path, VALUES, 'exported_cy'))
    assert compiled['installed'] == {'_c_Projection': True}
    for a, b in zip(compiled['second'], live['second']):
        assert all(_close(x, y) for x, y in zip(a, b))
    assert _close(compiled['collect'], live['collect']) and _close(compiled['direct'], live['direct'])
    assert all(_close(x, y) for x, y in zip(compiled['after_clear'], live['after_clear']))
    assert all(_close(x, y) for x, y in zip(compiled['subclass'], live['subclass']))


def test_threads_construct_release_and_collect_while_gc_runs(tmp_path):
    _build(tmp_path)
    assert 'errors []' in _run(tmp_path, STRESS)


PICKLING = textwrap.dedent('''\
    import copy, ctypes, gc, json, pickle
    from exported_cy import mx_model as m
    import exported_cy._mx_classes as classes
    p = m.Projection
    keys = list(p.keys)
    layout = classes._mx_lazy_values_layout['_c_Projection']

    def unset_nonzero(item):
        """Unset value slots whose raw bytes are not zero (read in place)."""
        bad, address = 0, id(item)
        for name, value, size, element, flag, flag_element in layout:
            n = size // element
            flags = ctypes.string_at(address + flag, n * flag_element)
            values = ctypes.string_at(address + value, size)
            for j in range(n):
                if not any(flags[j * flag_element:(j + 1) * flag_element]) \\
                        and any(values[j * element:(j + 1) * element]):
                    bad += 1
        return bad

    def dirty():
        for k in keys[:40]:
            p[k].value()
        for k in keys[:40]:
            del p[k]
        gc.collect()

    Sub = type('Sub', (type(p),), {})
    entries = {
        '__reduce_cython__': lambda x: x.__reduce_cython__(),
        '__reduce__': lambda x: x.__reduce__(),
        '__reduce_ex__(2)': lambda x: x.__reduce_ex__(2),
        '__reduce_ex__(4)': lambda x: x.__reduce_ex__(4),
        'copy.copy': copy.copy,
        'copy.deepcopy': copy.deepcopy,
        'pickle.dumps': lambda x: pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL),
    }
    out = {}
    for index, (name, call) in enumerate(entries.items()):
        dirty()
        item = p[keys[60 + index]]
        item.rate(3)
        before = unset_nonzero(item)
        try:
            call(item)
            outcome = 'ok'
        except Exception as error:      # the zeroing runs before any serialisation error
            outcome = type(error).__name__
        out[name] = [before, unset_nonzero(item), outcome]
    sub = Sub(p._parent)                # generic allocation: zero from the start
    sub._mx_copy_refs(p, p)
    sub.base_rate(3)
    out['subclass __reduce_ex__(4)'] = [unset_nonzero(sub), (sub.__reduce_ex__(4), unset_nonzero(sub))[1], 'ok']
    item = p[keys[100]]
    item.value()
    try:
        again = pickle.loads(pickle.dumps(item))
        out['round trip'] = again.value() == item.value()
    except Exception as error:
        out['round trip'] = type(error).__name__
    print(json.dumps(out))
    ''')


def test_every_pickling_entry_point_emits_zero_for_unset_slots(tmp_path):
    _build(tmp_path)
    generated = (tmp_path / 'exported_cy/_mx_classes.c').read_text(encoding='utf-8')
    call = re.findall(r'mx_zero_unset_\w*_mxv__c_Projection\(\(PyObject \*\)__pyx_v_self\);', generated)
    assert len(call) == 1
    reduce = re.search(r'static PyObject \*__pyx_pf_\w*?13_c_Projection_\d+__reduce_cython__\(struct '
                       r'__pyx_obj_\w*?__c_Projection \*__pyx_v_self\) \{\n  (.*)', generated)
    assert reduce and reduce.group(1) == call[0]
    out = json.loads(_run(tmp_path, PICKLING))
    for name, value in out.items():
        if name == 'round trip':
            continue
        before, after, outcome = value
        assert after == 0, (name, value)
    assert any(value[0] > 0 for name, value in out.items() if name not in ('round trip',)
               and not name.startswith('subclass')), out       # the memory was dirty: not vacuous
    assert out['round trip'] in (True, 'TypeError', 'PicklingError'), out


def test_the_build_fails_when_the_reduce_function_is_not_found_exactly_once(tmp_path):
    import pytest
    from modelx_cython import lazy_values
    head = 'static int mx_install_pkg_mod_mxv__c_A(void *t) {\n}\nx = mx_install_pkg_mod_mxv__c_A(((void *)t));\n'
    reduce = ('static PyObject *__pyx_pf_3pkg_4_c_A_8__reduce_cython__('
              'struct __pyx_obj_3pkg__c_A *__pyx_v_self) {\n  int x;\n}\n')
    missing, twice, once = tmp_path / 'missing.c', tmp_path / 'twice.c', tmp_path / 'once.c'
    missing.write_text(head, encoding='utf-8')
    twice.write_text(head + reduce + reduce, encoding='utf-8')
    once.write_text(head + reduce, encoding='utf-8')
    for path in (missing, twice):
        with pytest.raises(RuntimeError, match='exactly one generated __reduce_cython__'):
            lazy_values.patch_reduce(path)
    assert lazy_values.patch_reduce(once) == ['pkg_mod_mxv__c_A']
    text = once.read_text(encoding='utf-8')
    assert '{\n  mx_zero_unset_pkg_mod_mxv__c_A((PyObject *)__pyx_v_self);\n  int x;' in text
    lazy_values.patch_reduce(once)                      # idempotent
    assert text == once.read_text(encoding='utf-8')


def test_the_generated_module_compiles_without_warnings(tmp_path):
    # the allocator takes field offsets on a real instance (mx_probe_instance): an address taken
    # through None pointed outside that object, and gcc reported it as -Warray-bounds
    output = _build(tmp_path)
    warnings = [line for line in output.splitlines() if '_mx_classes.c:' in line and 'warning:' in line]
    assert not any('Warray-bounds' in line for line in output.splitlines()), warnings[:5]
    assert not warnings, warnings[:5]
