"""Items of a shared Space, held by each worker's own items, read from four threads at once.

Each Policy item holds a Market item of a locked (shared) Space in a zero-argument cached Cell and reads it
every period (``market().rate(...)``, ``market().price(...)``). Values at four threads equal one thread
and the export. After the run every Market item has the reference count of an
item no Policy item holds; and a thread that deletes the Market items from their Space while the workers
read them (the shared Space emptied and refilled), or a model instance dropped after its run, changes no value.
"""
import re
import subprocess
import sys
import textwrap

import modelx as mx


def test_held_items_of_a_shared_space_from_four_threads(tmp_path):
    model = mx.new_model('SharedHeldItems')
    try:
        market = model.new_space('Market', formula='lambda shock=-1: None')
        market.new_cells('price', 'lambda t: 1.0 / (1.0 + 0.03 + 0.0001 * shock) ** (t / 12.0)')
        market.new_cells('rate', 'lambda tenor, t: (price(t) / price(t + tenor)) ** (12.0 / tenor) - 1.0')
        keys = tuple(range(1, 201))
        policy = model.new_space('Policy', formula='lambda key: None',
                                 refs={'Market': market, 'months': tuple(range(60))})
        policy.new_cells('market', 'lambda: Market(key % 3 - 1)')
        policy.new_cells('tenor', 'lambda t: 12 + (key + t) % 49')
        policy.new_cells('cf', 'lambda t: key * 0.5 * (1.0 + market().rate(tenor(t), t))')
        policy.new_cells('discount', 'lambda t: market().price(t + 1)')
        policy.new_cells('pv', '''def pv():
    total = 0.0
    for t in months:
        total += cf(t) * discount(t)
    return total
''')
        model.export(tmp_path / 'exported', locked_spaces=['Market'])
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nfor k in (1, 2, 3):\n    m.Policy[k].pv()\n', encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'spaces': {
        'Market': {'cells_param_size': {'t': 128, 'tenor': 64}},
        'Policy': {'cells_param_size': {'t': 64}}}}), encoding='utf-8')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                            '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    # The Policy item's cache owns the Market item through both native calls.
    assert len(re.findall(r'_mx_cy\.cast\(_c_Market, _mx_pin_held_item_\d+\(self\)\)', source)) >= 2, source
    assert re.search(r'def _mx_pin_held_item_\d+\(root: _c_Policy\) -> _mx_cy\.pointer\(_mx_cy\.void\)', source)
    probe = textwrap.dedent('''\
        import gc
        import sys
        import threading
        from concurrent.futures import ThreadPoolExecutor
        m = __import__(sys.argv[1]).mx_model
        KEYS = range(1, 201)

        def run(model, workers, churn=False):
            P, M = model.Policy, model.Market
            def value(k):
                v = P[k].pv()
                del P[k]
                return v
            done = threading.Event()
            def delete_markets():          # the shared Space emptied and refilled under the readers
                while not done.is_set():
                    for shock in (-1, 0, 1):
                        try:
                            del M[shock]
                        except KeyError:
                            pass
            deleter = threading.Thread(target=delete_markets) if churn else None
            if deleter:
                deleter.start()
            try:
                if workers == 1:
                    return [value(k) for k in KEYS]
                with ThreadPoolExecutor(workers) as pool:
                    return list(pool.map(value, KEYS, chunksize=7))
            finally:
                done.set()
                if deleter:
                    deleter.join()

        out = []
        for workers, churn in ((1, False), (4, False), (4, True), (4, False), (4, True)):
            fresh = type(m)()             # a compiled model has no clear: a new instance per run
            out.append(run(fresh, workers, churn))
            gc.collect()
            if sys.argv[1] == 'exported_cy' and not churn:
                held = [fresh.Market(s) for s in (-1, 0, 1)]
                unheld = fresh.Market(5)
                counts = {sys.getrefcount(item) for item in held + [unheld]}
                assert len(counts) == 1, counts      # no Policy item holds, or leaked, a reference
                del held, unheld
            del fresh                     # the model instance and its shared Space dropped
            gc.collect()
        print(repr(out))
        ''')
    values = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True,
                             timeout=600)
        assert run.returncode == 0, run.stderr
        values[package] = eval(run.stdout)
    serial = values['exported'][0]
    assert all(run == serial for run in values['exported'])
    assert all(run == serial for run in values['exported_cy']), [
        max(abs(a - b) for a, b in zip(run, serial)) for run in values['exported_cy']]
