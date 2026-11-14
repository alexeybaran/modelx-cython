"""W1.2 typed handles beyond the matrix: binding, sharing, six-level keys, counted fallbacks.

Uses the matrix session build (table_matrix_cases_w1_2.build): a compiled model whose Data
cells build their tables through the compiled pdn.table, and the handle driver.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

from modelx_helpers import pdn
from table_matrix_cases_w1_2 import build


@pytest.fixture(scope='module')
def b(tmp_path_factory):
    return build(tmp_path_factory)


def _provider(b):
    return importlib.import_module(b.provider)


def test_handle_is_built_once_per_table_and_bound(b):
    provider = _provider(b)
    first, second = b.compiled.base('R3'), b.compiled.base('R3')
    assert type(first).__name__ == 'NativeTable' and first.handle is not None
    owner = first.__modelx_table_owner__()
    assert owner._native_cache[provider.TableHandle] is first.handle      # set at preparation
    assert second.handle is first.handle
    assert provider.table_handle_of(first) is first.handle
    rebound = provider.NativeTable(owner) if hasattr(provider, 'NativeTable') else None
    if rebound is not None:
        assert rebound.handle is first.handle                          # a later bind finds it


def test_children_are_shared_views(b):
    d, native = b.driver, b.compiled.base('F2')
    h = native.handle
    assert h.scheme_name == 'hashed' and h.shape == (6, 2, 3, 0)
    one = d.sub(native, (1,), True)
    assert one is d.sub(native, (1,), True)                              # eager, not rebuilt
    assert one in h.children and one.prefix == (1,)
    assert d.sub(native, (1, 'y'), False) is d.sub(one, ('y',), False)
    assert one.helper() is native.__modelx_table_owner__().loc[1]     # the helper's own memo child
    r = d.sub(b.compiled.base('R3'), (0,), True)
    assert r.scheme_name == 'product' and r.shape[3] == 6                 # a view at row 6
    c = d.sub(r, ('x',), False)
    assert c.scheme_name == 'dense' and c.shape == (3, 1, 1, 9)
    memory = h.memory()
    assert memory['handles'] == 1 + 2 + 4                                  # root, 2 + 4 prefixes


def test_six_level_keys(b):
    provider, d = _provider(b), b.driver
    index = pd.MultiIndex.from_tuples([(a, 'x', c, 'y', e, f) for a in (0, 1) for c in (2, 3)
                                       for e in (4, 5) for f in (6, 7)])
    frame = pd.DataFrame({'q': np.arange(len(index), dtype=float)}, index=index)
    h = provider.TableHandle(pdn.table(frame))
    for position, key in enumerate(index):
        assert d.loc(h, key, 'q', False) == position
        ints = tuple(part for part in key if type(part) is int)
        assert d.contains(h, key, False)
    assert d.loc(d.sub(h, (1, 'x', 3, 'y', 5), False), (6,), 'q', True) == frame.loc[(1, 'x', 3, 'y', 5, 6), 'q']
    assert d.loc(d.sub(h, (1, 'x', 3, 'y'), False), (5, 7), 'q', True) == frame.loc[(1, 'x', 3, 'y', 5, 7), 'q']
    assert d.contains(h, (1, 'x', 3, 'y', 5), False) and not d.contains(h, (1, 'x', 3, 'z', 5), False)
    with pytest.raises(KeyError) as error:
        d.loc(h, (1, 'x', 3, 'y', 5, 8), 'q', False)
    assert error.value.args == ((1, 'x', 3, 'y', 5, 8),)
    with pytest.raises(KeyError) as error:
        d.sub(h, (1, 'x', 9), False)
    assert error.value.args == ((1, 'x', 9),)
    all_int = pd.DataFrame({'q': np.arange(64.0)},
                           index=pd.MultiIndex.from_product([[0, 1]] * 6))
    g = provider.TableHandle(pdn.table(all_int))
    assert g.scheme_name == 'product'
    assert d.loc(g, (1, 0, 1, 0, 1, 1), 'q', True) == 43.0
    assert d.contains(g, (1, 0, 1, 0, 1), True) and not d.contains(g, (1, 0, 2), True)
    assert d.loc(d.sub(g, (1, 0, 1, 0, 1), True), (1,), 'q', True) == 43.0


def test_python_fallbacks_are_counted(b):
    provider, d = _provider(b), b.driver
    source = pdn.table(pd.DataFrame({'q': [1.0, 2.0, 3.0]}, index=[30, 10, 20]))
    unordered = provider.TableHandle(source)
    assert unordered.scheme_name == 'sorted'
    before = provider.handle_fallbacks()
    assert d.loc(unordered, (10,), 'q', True) == 2.0
    for x in (5, 10, 15, 25, 35):                                     # bisect in row order, as the helper
        assert d.searchsorted(unordered, x) == source.index.searchsorted(x, side='right')
    assert provider.handle_fallbacks() == before                     # native
    record = d.row_record(b.compiled.base('G'), (20,), True)          # a row that escapes
    assert record['n'] == 2 and provider.handle_fallbacks() == before + 1
    with pytest.raises(Exception):
        d.asof(unordered, 25)                                         # pandas decides (not monotonic)
    assert provider.handle_fallbacks() == before + 2


def test_unbound_tables_and_records(b):
    provider, d = _provider(b), b.driver
    table = pdn.table(pd.DataFrame({'q': [1.5, 2.5]}, index=[4, 5]))
    before = provider.handle_fallbacks()
    assert d.loc(table, (5,), 'q', True) == 2.5                       # helper Table: found by cache
    assert provider.handle_fallbacks() == before + 1
    handle = provider.table_handle_of(table)
    assert handle is table._native_cache[provider.TableHandle]
    record = b.compiled.point()
    before = provider.handle_fallbacks()
    assert d.field(record, 'q') == 0.75 and d.field_int(record, 'n') == 1
    assert d.field_obj(record, 's') == 'MALE'
    assert provider.handle_fallbacks() == before
    with pytest.raises(KeyError) as error:
        d.field(record, 'missing')
    assert error.value.args == ('missing',)


def test_table_object_of_a_sub_table(b):
    import pickle
    provider, d = _provider(b), b.driver
    native = b.compiled.base('F2')
    owner = native.__modelx_table_owner__()
    child = d.sub(native, (1,), True)
    table = d.sub_object(native, (1,), True)
    assert table is d.sub_object(native, (1,), True) and table.handle is child
    cached = dict(owner._native_cache.get(('prefix_children', 1), ({},))[0])
    before = provider.handle_fallbacks()
    assert d.loc(table, ('y', 2), 'q', False) == 6.0 and provider.handle_fallbacks() == before
    assert dict(owner._native_cache.get(('prefix_children', 1), ({},))[0]) == cached   # no helper child made
    assert table.loc[('y', 2), 'q'] == 6.0                           # Python use: the helper child
    assert table.__modelx_table_owner__() is owner.loc[1]
    assert ('y', 1) in table.index and len(table.index) == 3
    restored = pickle.loads(pickle.dumps(table))
    assert restored.loc[('x', 3), 'b'] == 1.5
    root = d.sub_object(b.compiled.base('R3'), (0, 'x'), False)
    assert root.handle.scheme_name == 'dense' and root.index.max() == 7
    whole = native.handle
    assert d.loc(whole, (1, 'y', 2), 'q', False) == 6.0


def test_sub_table_object_holds_its_helper_sub_table(b, monkeypatch):
    """Python reads on a sub-table's bound table (D53) select the helper sub-table once, not
    on every read: per-step formulas read `sub["c"]` / `sub.index` on a per-policy table-valued
    cell, and re-running `owner.loc[prefix]` per read doubled UC_FR_S (diag-uc-fr)."""
    from modelx_helpers.immutable_inputs import Table
    d, native = b.driver, b.compiled.base('F2')
    table = d.sub_object(native, (1,), True)
    first = table.__modelx_table_owner__()          # materialised here (the helper's memo child)
    assert first is native.__modelx_table_owner__().loc[1]
    calls = []
    loc = Table.loc
    monkeypatch.setattr(Table, 'loc', property(lambda self: (calls.append(self), loc.fget(self))[1]))
    for _ in range(3):
        assert table.__modelx_table_owner__() is first
        assert ('y', 1) in table.index and len(table.index) == 3 and len(list(table.index)) == 3
        assert list(table['q']) == list(first['q'])
    assert calls == []


def test_plain_table_handle_is_the_cached_one_from_every_thread(b):
    """diag-4w: the handle of a plain helper Table (a Data table not bound as NativeTable) is read
    from its native cache without the object's lock or reference counts on the shared cache; the
    result is the one cached handle, from every thread, and anything else raises as before."""
    import threading
    provider = _provider(b)
    table = pdn.table(pd.DataFrame({'q': [1.0, 2.0, 3.0]}, index=[1, 2, 3]))
    first = provider.table_handle_of(table)
    assert table._native_cache[provider.TableHandle] is first
    seen, errors = [], []

    def read():
        try:
            seen.extend(provider.table_handle_of(table) for _ in range(2000))
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=read) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and len(seen) == 8000 and all(handle is first for handle in seen)
    with pytest.raises(TypeError):
        provider.table_handle_of(object())
