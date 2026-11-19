"""Tables in compiled models behave exactly as on live modelx (T-1 probe; design toolkit p2/tables/DESIGN.md).

One model applies each family of pandas use to tables held by zero-argument Data cells: a flat frame (int
index), a two-level frame, a Series, a non-unique two-level frame (as Annuity_JP_S's mort_anchor_table), and the
same frame through the table library's ``pdn.table`` (only while that library exists). Every Calc cell has an
argument ``t`` so its key depends on the call. Each cell is compared with live modelx for t = 0..3 STRICTLY:
same type and value, or the same exception. ``KNOWN`` lists, by cell, how the compiled model differs today;
each step of the table redesign removes entries, and the final state is an empty ``KNOWN`` except the documented
boundary types (``BOUNDARY``: a C-typed cell result returns ``float``/``int``, an array cell a memoryview, as
upstream modelx-cython does).

Separately, ``test_reassign_and_clear`` checks the input-change contract agreed with the owner: assigning a new
frame to a Data cell and calling clear_all (on the dependent Space and on the model) gives live's results on the
new frame; without clear_all, cached results stay, but no prepared copy of the reassigned cell survives.
Model.clear_all discards an assigned input on live modelx (the cell recomputes its formula); compiled the same.
"""
import re
import json
import subprocess
import sys

import pytest

OPS = {  # name: (sources, expression on T = data.<source>() and the argument t)
    'loc_scalar': ('flat', "T.loc[20 + t, 'q']"),
    'loc_row': ('flat', "T.loc[20 + t]"),
    'loc_row_field': ('flat', "T.loc[20 + t]['q']"),
    'loc_missing': ('flat', "T.loc[99 + t, 'q']"),
    'loc_tuple': ('multi', "T.loc[(('A', 'B', 'C')[t % 3], 1 + t % 2), 'rate']"),
    'loc_prefix': ('multi', "T.loc[('A', 'B', 'C')[t % 3]]"),
    'loc_prefix_key': ('multi', "T.loc[('A', 'B', 'C')[t % 3]].loc[1 + t % 2, 'rate']"),
    'loc_prefix_col': ('multi', "T.loc[('A', 'B', 'C')[t % 3], 'rate']"),
    'loc_slice': ('flat', "T.loc[20:20 + t, 'q']"),
    'loc_list': ('flat', "T.loc[[20, 20 + t], 'q']"),
    'loc_mask': ('flat', "T.loc[T['q'] > 0.001 * t]"),
    'loc_callable': ('flat', "T.loc[lambda d: d.index > 20 + t]"),
    'at': ('flat', "T.at[20 + t, 'r']"),
    'at_tuple': ('multi', "T.at[(('A', 'B', 'C')[t % 3], 1 + t % 2), 'rate']"),
    'nonunique_key_col': ('dup', "T.loc[(('A', 'B')[t % 2], 1), 'rate'].iloc[0]"),
    'nonunique_rows': ('dup', "T.loc[(('A', 'B')[t % 2], 1)]"),
    'iloc': ('flat', "T.iloc[t, 0]"),
    'iloc_row': ('flat', "T.iloc[t]"),
    'iat': ('flat', "T.iat[t, 1]"),
    'col_attr': ('flat', "T.q"),
    'col_item': ('flat', "T['q']"),
    'col_keyed': ('flat', "T['q'][20 + t]"),
    'col_list': ('flat', "T[['q', 'r']]"),
    'col_attr_keyed': ('flat', "T.q[20 + t]"),
    'col_keyed_tuple': ('multi', "T['rate'][(('A', 'B', 'C')[t % 3], 1 + t % 2)]"),
    'index_in_prefix': ('multi grid', "('A', 'B', 'C', 'D')[t % 4] in T.index"),
    'index_in_prefix_tuple': ('rates', "(('A', 'B')[t % 2], ('F', 'X')[t // 2]) in T.index"),
    'index_max': ('flat multi', "T.index.max()"),
    'index_min': ('flat', "T.index.min()"),
    'index_max_part': ('multi', "T.index.max()[t % 2]"),
    'index_asof': ('flat', "T.index.asof(20.5 + t)"),
    'index_get_loc': ('flat', "T.index.get_loc(20 + t)"),
    'index_arith': ('flat', "T.index + t"),
    'index_cmp': ('flat', "T.index > 20 + t"),
    'index_list': ('flat', "list(T.index)"),
    'index_len': ('flat', "len(T.index) + t"),
    'index_in': ('flat multi', "(20 + t) in T.index"),
    'len': ('flat series', "len(T) + t"),
    'iter': ('flat', "list(T)"),
    'contains': ('flat', "'q' in T"),
    'arith': ('flat series', "T * (t + 1)"),
    'col_arith': ('flat', "T['q'] * (t + 1)"),
    'cmp': ('flat series', "T > 0.001 * t"),
    'sum': ('flat series', "T.sum()"),
    'col_mean': ('flat', "T['q'].mean() + t"),
    'max': ('flat series', "T.max()"),
    'values': ('flat series', "T.values"),
    'to_numpy': ('flat series', "T.to_numpy()"),
    'iterrows': ('flat', "[(k, r['q']) for k, r in T.iterrows()]"),
    'iterrows_sum': ('flat', "sum(r['q'] * (k - 19) for k, r in T.iterrows()) + t"),
    'itertuples': ('flat', "[tuple(x) for x in T.itertuples()]"),
    'groupby': ('multi', "T.groupby(level=0).sum()"),
    'merge': ('flat', "pd.DataFrame({'age': [20, 20 + t], 'w': [1.0, 2.0]}).merge(T, left_on='age', right_index=True)"),
    'join': ('flat', "pd.DataFrame({'age': [20, 20 + t]}).join(T, on='age')"),
    'apply': ('flat', "T.apply(sum)"),
    'transpose': ('flat', "T.T"),
    'pickle': ('flat', "pickle.loads(pickle.dumps(T))"),
    'isinstance': ('flat series', "isinstance(T, (pd.DataFrame, pd.Series))"),
    'type_name': ('flat multi series', "type(T).__name__"),
    'np_asarray': ('flat', "np.asarray(T)"),
    'np_exp': ('flat', "np.exp(T['q'])"),
    'pd_concat': ('flat', "pd.concat([T, T])"),
    'pd_frame': ('flat', "pd.DataFrame(T)"),
    'series_key': ('series', "T[20 + t]"),
    'series_loc': ('series', "T.loc[20 + t]"),
    'series_sum': ('series', "T.sum() + t"),
    'identity': ('flat', "T is data.flat()"),
    'mutate_loc': ('mut', "(T.loc.__setitem__((20, 'q'), 9.0 + t), T.loc[20, 'q'])[1]"),
}

# T-5: formulas of several statements on T = data.<source>() and t (views of a prefix, index walks, row locals)
STMTS = {
    'view_index': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\n"
                   "return (int(sub.index[0]), int(sub.index[-1]), sub.index[t % 4], sub.index[-1 - t], len(sub.index))"),
    'view_search': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\nx = 20.5 + t * 0.7\n"
                    "return (sub.index.searchsorted(x, side='right'), sub.index.searchsorted(x), int(sub.index.asof(x)))"),
    'view_asof_low': ('grid', "return int(T.loc['A'].index.asof(19.0 + t))"),
    'view_asof_key': ('grid', "sub = T.loc['B']\nreturn sub.loc[sub.index.asof(19.5 + t), 'q']"),
    'view_read': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\nreturn sub.loc[20 + t, 'q'] + sub.loc[23 - t, 'n']"),
    'view_walk': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\nreturn [sub.loc[k, 'q'] for k in sub.index if k <= 21 + t]"),
    'view_loop': ('grid sgrid', "sub = T.loc['B']\ntotal = 0.0\nfor k in sub.index:\n"
                  "    if k > 22 - t:\n        break\n    total += sub.loc[k, 'q'] * k\nreturn total"),
    'view_builtins': ('grid sgrid', "sub = T.loc['A']\nreturn (sorted(sub.index), list(sub.index), set(sub.index), "
                      "tuple(sub.index), min(sub.index), max(sub.index), sub.index.max() + t, sub.index.min())"),
    'whole_walk': ('flat shuffled', "return [T.loc[k, 'q'] for k in T.index if k > 19 + t]"),
    'whole_loop': ('flat shuffled', "out = []\nfor k in T.index:\n    out.append(k * 2)\nreturn out[t % 4]"),
    'whole_item': ('flat shuffled', "return (T.index[t % 4], T.index[-1 - t % 4], T.index.searchsorted(21 + t), "
                   "len(T.index), list(T.index), sorted(T.index))"),
    'whole_unpack': ('grid sgrid', "return (max(a for p, a in T.index if p == ('A', 'B')[t % 2]), "
                     "[(p, a) for p, a in T.index][t])"),
    'row_local': ('mixed', "row = T.loc[20 + t]\nreturn (row['q'], row['tag'], row['n'], str(row['tag']))"),
    'row_local_num': ('flat', "row = T.loc[20 + t]\nreturn row['q'] + row['r']"),
    'row_local_missing': ('flat', "row = T.loc[22 + t]\nreturn row['q']"),
    'row_in_loop': ('mixed', "total = 0.0\nfor k in T.index:\n    row = T.loc[k]\n"
                    "    if str(row['tag']) != 'x':\n        continue\n    total += float(row['q']) * row['n']\nreturn total + t"),
    'extreme_obj': ('fgrid grid', "return (T.index.max(), T.index.min(), t)"),
    'inline_max': ('grid sgrid', "return int(T.loc[('A', 'B')[t % 2]].index.max()) + t"),
    'nested_key': ('flat', "return T.loc[min(20 + t, int(T.index.max())), 'q']"),
    'two_part_prefix': ('tri', "sub = T.loc[('A', 'B')[t % 2], ('F', 'M')[t // 2]]\nreturn (list(sub.index), sub.loc[21, 'q'])"),
    'two_part_prefix_amb': ('tri_amb', "sub = T.loc[('A', 'B')[t % 2], ('q', 'x')[t // 2]]\n"
                            "return (list(sub.index), sub.loc[21, 'q'])"),
    'index_in_key_local': ('multi', "key = (('A', 'B', 'D')[t % 3], 1 + t % 2)\nreturn (key in T.index, key not in T.index)"),
    'named_col': ('flat', "name = ('q', 'r', 'n')[t % 3]\nreturn T.loc[20 + t, name]"),
    'named_col_mixed': ('grid', "name = ('q', 'n')[t % 2]\nreturn T.loc[('A', 20 + t), name]"),
    'named_col_float': ('mixed', "name = ('q', 'n', 'tag')[t % 3]\nreturn float(T.loc[20 + t, name])"),
    'named_col_fresh': ('flat mixed', "name = ''.join([('q', 'r', 'n')[t % 3]])\nreturn float(T.loc[20 + t, name])"),
    'named_col_missing': ('flat', "return T.loc[20, ('q', 'zz')[t % 2]]"),
    'named_col_view': ('tri', "sub = T.loc['A', ('F', 'M')[t % 2]]\nreturn sub.loc[20 + t % 2, ('q', 'x')[t // 2]]"),
    'col_view_items': ('grid sgrid', "tab = T.loc[('A', 'B')[t % 2]]['q']\nreturn [(float(k), float(v)) for k, v in tab.items()]"),
    'col_view_iter': ('grid', "return tuple(float(p) for p in T.loc['B']['n'])"),
    'col_view_tolist': ('grid sgrid', "sub = T.loc['A']\nreturn list(zip(sub['q'].tolist(), sub['n'].tolist(), [len(sub['q'])]))"),
    'col_whole': ('flat shuffled', "return ([v for v in T['q']], T['r'].tolist(), list(T['q']), "
                  "[k for k, v in T['q'].items() if v > 0.001 * t])"),
    'col_loop': ('flat shuffled', "total = 0.0\nfor k, v in T['q'].items():\n    total += k * v\nreturn total + t"),
    'iterrows_loop': ('flat shuffled mixed', "total = 0.0\nfor k, row in T.iterrows():\n"
                      "    if k > 22 - t:\n        continue\n    total += row['q'] * k + float(row[('q', 'q')[t % 2]])\nreturn total"),
    # loop targets the lowering does not take: each declines to pandas (no compiler error)
    'for_three': ('flat', "total = 0.0\nfor k, q, r in zip(T.index, T['q'], T['r']):\n    total += k * q + r\nreturn total + t"),
    'for_nested': ('flat', "total = 0.0\nfor k, (q, r) in zip(T.index, zip(T['q'], T['r'])):\n"
                   "    total += k * q - r\nreturn total + t"),
    'iterrows_nested': ('flat', "total = 0.0\nfor k, (q, r) in T[['q', 'r']].iterrows():\n    total += k * q + r\n"
                        "return total + t"),
    'iterrows_star': ('flat', "total = 0.0\nfor k, *rest in T.iterrows():\n    total += k * rest[0]['q']\nreturn total + t"),
    'iterrows_three': ('flat', "total = 0.0\nfor i, (k, row) in enumerate(T.iterrows()):\n    total += i * row['q'] + k\n"
                       "return total + t"),
    'row_slice': ('flat', "row = T.loc[20 + t]\nreturn float(row['q']) + float(row[:1].iloc[0])"),
    'iterrows_find': ('mixed', "for k, row in T.iterrows():\n    if int(row['n']) >= 1 + t:\n        return (k, str(row['tag']))\nreturn None"),
    'obj_col': ('mixed', "return (T.loc[20 + t, 'tag'], T.at[20 + t, 'tag'] == 'x')"),
    'obj_col_missing': ('mixed', "return T.loc[98 + t, 'tag']"),
    'view_within': ('scen', "sub = T.loc[('up', 'down')[t % 2]]\nmonths = [m for m in sub.index if m <= 6 * t]\n"
                    "return (sub.loc[max(months), 'v'], int(sub.index.asof(5.0 * t)), sub.index[-1])"),
    'level_extreme': ('grid sgrid tri', "return (int(T.index.get_level_values(1).max()) + t, T.index.get_level_values(-1).min())"),
    'level_extreme_hex': ('grid', "return int(T.index.get_level_values(0x1).max()) + t"),
    'level_extreme_named': ('rates', "return (int(T.index.get_level_values('year').max()) - t, T.index.get_level_values('year').min())"),
    'col_of_loc': ('tri', "curve = T.loc[(('A', 'B')[t % 2], ('F', 'M')[t // 2]), 'q']\nms = [int(v) for v in curve.index]\n"
                   "return (ms, float(curve.iloc[0]), float(curve.iloc[-1]), float(curve.loc[21]))"),
    'whole_asof': ('flat shuffled', "return int(T.index.asof(21.5 + t))"),
    'view_member': ('grid sgrid', "sub = T.loc[('A', 'B')[t % 2]]\nreturn ((20 + t) in sub.index, (21 + 2 * t) not in sub.index)"),
}

DATA = {
    'flat': "pd.DataFrame({'q': [0.001, 0.002, 0.004, 0.008], 'r': [1.0, 2.0, 3.0, 4.0]},\n"
            "                        index=pd.Index([20, 21, 22, 23], name='age'))",
    'multi': "pd.DataFrame({'rate': [1.5, 1.6, 2.5, 2.6, 3.5, 3.6]},\n"
             "        index=pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1), ('B', 2), ('C', 1), ('C', 2)],\n"
             "                                        names=['plan', 'year']))",
    'series': "pd.Series([0.001, 0.002, 0.004, 0.008], index=pd.Index([20, 21, 22, 23], name='age'), name='q')",
    'mut': "pd.DataFrame({'q': [0.001, 0.002, 0.004, 0.008], 'r': [1.0, 2.0, 3.0, 4.0]},\n"
           "                        index=pd.Index([20, 21, 22, 23], name='age'))",
    'dup': "pd.DataFrame({'rate': [1.5, 1.7, 2.5, 2.7]},\n"
           "        index=pd.MultiIndex.from_tuples([('A', 1), ('A', 1), ('B', 1), ('B', 1)], names=['tab', 'sex']))",
    # T-5: a two-level table in sorted order, the same rows unordered, a float index, a row with an object column
    'grid': "pd.DataFrame({'q': [0.01 * n for n in range(8)], 'n': list(range(8))},\n"
            "        index=pd.MultiIndex.from_product([['A', 'B'], [20, 21, 22, 23]], names=['plan', 'age']))",
    'sgrid': "pd.DataFrame({'q': [0.01 * n for n in range(8)], 'n': list(range(8))},\n"
             "        index=pd.MultiIndex.from_product([['A', 'B'], [20, 21, 22, 23]], names=['plan', 'age'])).iloc[[5, 0, 7, 2, 1, 4, 3, 6]]",
    'shuffled': "pd.DataFrame({'q': [0.004, 0.001, 0.008, 0.002], 'r': [3.0, 1.0, 4.0, 2.0]},\n"
                "                        index=pd.Index([22, 20, 23, 21], name='age'))",
    'fgrid': "pd.DataFrame({'q': [0.1, 0.2, 0.3]}, index=pd.Index([0.5, 1.5, 2.5], name='x'))",
    'mixed': "pd.DataFrame({'q': [0.5, 1.5, 2.5, 3.5], 'tag': ['x', 'y', 'x', 'y'], 'n': [1, 2, 3, 4]},\n"
             "                        index=pd.Index([20, 21, 22, 23], name='age'))",
    # T-5: two scenarios out of sorted order, each in sorted order within (ordered within a prefix, not as a whole)
    'scen': "pd.DataFrame({'v': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},\n"
            "        index=pd.MultiIndex.from_tuples([('up', 0), ('up', 12), ('up', 24), ('down', 0), ('down', 12), ('down', 24)]))",
    # T-5 choice 5: `T.loc[a, b]` on three levels, the labels of b's level apart from the columns, or among them
    'tri': "pd.DataFrame({'q': [0.1 * n for n in range(8)], 'x': [0.0] * 8},\n"
           "        index=pd.MultiIndex.from_product([['A', 'B'], ['F', 'M'], [20, 21]]))",
    'tri_amb': "pd.DataFrame({'q': [0.1 * n for n in range(8)], 'x': [0.0] * 8},\n"
               "        index=pd.MultiIndex.from_product([['A', 'B'], ['q', 'x'], [20, 21]]))",
}

# T-5: item cells (a view of a prefix fixed per item: codes held per item; walks of a shared table)
T5_ITEMS = {
    'view_item': "sub = data.grid().loc[plan()]\nreturn sub.loc[20 + t, 'q'] + int(sub.index[-1]) + len(sub.index)",
    'view_item_walk': "sub = data.grid().loc[plan()]\nreturn sum(sub.loc[k, 'q'] for k in sub.index if k <= 20 + t)",
    'whole_item_walk': "return [k for k in data.shuffled().index if k > 20 + t]",
    'row_item': "row = data.mixed().loc[20 + t]\nreturn row['q'] * row['n']",
    # a row cell kept (one read takes a computed field): its fields through the row held per item
    'row_cell_lit': "return mrow()['q'] * mrow()['n']",
    'obj_item': "return data.mixed().loc[pid(), 'tag'] + data.mixed().loc[20 + t, 'tag']",
    'row_cell_computed': "return float(mrow()[('q', 'n')[t % 2]]) + float(mrow()['q'])",
}

# Compiled differs from live, by cell: what the redesign removes step by step. Empty since T-3 (the sample
# runs on the user's frames; nothing stands in for them). The 13 entries at 84befe4 are in the T-1 report.
KNOWN = {}

# Boundary types that equal upstream modelx-cython's (a C-typed cell result, an array-valued cell)
BOUNDARY = {'float for numpy.float64', 'int for numpy.int64', 'memoryview for ndarray', 'vector for list'}

MODEL = '''\
import modelx as mx
import numpy as np
import pandas as pd
import pickle
OPS = {ops!r}
STMTS = {stmts!r}
DATA = {data!r}
T5_ITEMS = {t5_items!r}


def names():
    for op, (sources, _) in (*OPS.items(), *STMTS.items()):
        for source in sources.split():
            yield f"{{op}}__{{source}}"


def _indent(body):
    return "".join(f"    {{line}}\\n" for line in body.split("\\n"))


def build():
    model = mx.new_model("Probe")
    data = model.new_space("Data")
    for name, expression in DATA.items():
        data.new_cells(name, f"def {{name}}():\\n    return {{expression}}\\n")
    data.pd = pd
    calc = model.new_space("Calc")
    calc.data, calc.pd, calc.np, calc.pickle = data, pd, np, pickle
    for op, (sources, expression) in OPS.items():
        for source in sources.split():
            calc.new_cells(f"{{op}}__{{source}}",
                           f"def {{op}}__{{source}}(t):\\n    T = data.{{source}}()\\n    return {{expression}}\\n")
    for op, (sources, body) in STMTS.items():
        for source in sources.split():
            calc.new_cells(f"{{op}}__{{source}}", f"def {{op}}__{{source}}(t):\\n    T = data.{{source}}()\\n" + _indent(body))
    items = model.new_space("Items", formula="lambda i: None")     # an ItemSpace: reads held per item
    items.data = data
    items.new_cells("v__flat", "def v__flat(t):\\n    return data.flat().loc[20 + t, 'q']\\n")
    items.new_cells("in__multi", "def in__multi(t):\\n    return (('A', 'B', 'C', 'D')[t % 4], 1) in data.multi().index\\n")
    # T-4: a prefix cell (traced: the sub-frame of Data.rates), its reads, a rejected prefix shape, keys in locals
    data.new_cells("rates", "def rates():\\n    return pd.DataFrame({{'q': [0.01 * n for n in range(12)]}}, index="
                   "pd.MultiIndex.from_product([['A', 'B'], ['F', 'M'], [2000, 2001, 2002]], names=['plan', 'sex', 'year']))\\n")
    items.new_cells("plan", "def plan():\\n    return ('A', 'B')[i % 2]\\n")
    items.new_cells("sex", "def sex():\\n    return ('F', 'M')[(i // 2) % 2]\\n")
    items.new_cells("sched", "def sched():\\n    return data.rates().loc[(plan(), sex())]\\n")
    items.new_cells("pre__rates", "def pre__rates(t):\\n    sub = sched()\\n    return sub.loc[2000 + t % 3, 'q']\\n")
    items.new_cells("sched_list", "def sched_list():\\n    return data.rates().loc[[plan()]]\\n")
    items.new_cells("pre_list__rates", "def pre_list__rates(t):\\n    return sched_list().loc[(plan(), sex(), 2000 + t % 3), 'q']\\n")
    items.new_cells("key_local__rates", "def key_local__rates(t):\\n    key = (plan(), sex(), 2000 + t % 3)\\n"
                    "    return data.rates().loc[key, 'q']\\n")
    items.new_cells("key_rebound__rates", "def key_rebound__rates(t):\\n    key = (plan(), sex(), 2000)\\n"
                    "    if t % 2:\\n        key = (plan(), sex(), 2001)\\n    return data.rates().loc[key, 'q']\\n")
    items.new_cells("pid", "def pid():\\n    return 20 + i % 4\\n")
    items.new_cells("mrow", "def mrow():\\n    return data.mixed().loc[pid()]\\n")
    for name, body in T5_ITEMS.items():
        items.new_cells(name, f"def {{name}}(t):\\n" + _indent(body))
    return model


if __name__ == "__main__":
    build().export("probe_nomx", locked_spaces=["Data"])
'''

SAMPLE = '''\
from probe_nomx import mx_model
from probe_model import names
for name in names():
    for t in range(4):
        try:
            getattr(mx_model.Calc, name)(t)
        except Exception:
            pass
from probe_model import T5_ITEMS
ITEM_CELLS = ("v__flat", "in__multi", "pre__rates", "pre_list__rates", "key_local__rates", "key_rebound__rates",
              *T5_ITEMS)
for i in (1, 2, 3, 4):
    for t in range(4):
        for name in ITEM_CELLS:
            getattr(mx_model.Items[i], name)(t)
'''

CHECKS = '''\
import json
import sys
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
import numpy as np
import pandas as pd
from probe_model import build, names, T5_ITEMS

assert isinstance(import_module("probe_nomx_cy._mx_classes").__loader__, ExtensionFileLoader)
compiled = import_module("probe_nomx_cy").mx_model
live = build()


def outcome(space, name, t):
    try:
        return "ok", getattr(space, name)(t)
    except Exception as error:
        return "raise", error


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, (pd.DataFrame, pd.Series, pd.Index)):
        return a.equals(b) and (not isinstance(a, pd.DataFrame) or list(a.columns) == list(b.columns))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and bool((a == b).all())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and a != a:
        return b != b
    return bool(a == b)


def boundary(a, b):
    """The documented boundary types: equal value, the type a C-typed or array cell gives Python."""
    kinds = {(np.float64, float): "float for numpy.float64", (np.int64, int): "int for numpy.int64"}
    kind = kinds.get((type(a), type(b)))
    if kind and a == b:
        return kind
    if isinstance(a, np.ndarray) and type(b).__name__ == "_memoryviewslice" and np.array_equal(a, np.asarray(b)):
        return "memoryview for ndarray"
    if isinstance(a, list) and type(b).__name__ in ("I64Vec", "F64Vec") and list(b) == a:
        return "vector for list"
    if isinstance(a, (list, tuple)) and type(a) is type(b) and len(a) == len(b):
        parts = [boundary(x, y) if not same(x, y) else "" for x, y in zip(a, b)]
        if all(p is not None for p in parts):
            found = {p for p in parts if p}
            return "; ".join(sorted(found)) if found else ""     # each element a documented boundary
    return None


result = {}
cases = [(name, name, live.Calc, compiled.Calc) for name in names()] + [
    (f"Items[{{i}}].{{name}}", name, live.Items[i], compiled.Items[i]) for name in T5_ITEMS for i in (1, 2, 3, 4)]
for key, name, live_space, compiled_space in cases:
    verdict = "same"
    for t in range(4):
        (lk, lv), (ck, cv) = outcome(live_space, name, t), outcome(compiled_space, name, t)
        if lk == "raise" and ck == "raise":
            original = lv.__cause__ or lv
            if type(cv) is type(original) or type(cv).__name__ + ":" in str(lv):
                continue
            verdict = f"other error: live {type(lv).__name__}, compiled {type(cv).__name__}: {str(cv)[:120]}"
        elif ck == "raise":
            verdict = f"raises {type(cv).__name__}: {str(cv)[:120]}"
        elif lk == "raise":
            verdict = f"returns {type(cv).__name__} where live raises {type(lv).__name__}"
        elif not same(lv, cv):
            kind = boundary(lv, cv)
            verdict = f"boundary: {kind}" if kind else f"differs: live {type(lv).__name__} {repr(lv)[:60]}, compiled {type(cv).__name__} {repr(cv)[:60]}"
        if verdict != "same":
            break
    result[key] = verdict
json.dump(result, sys.stdout, indent=1)
'''


CONTRACT = '''\
import json
import re
import sys
from importlib import import_module
import pandas as pd
from probe_model import build

compiled = import_module("probe_nomx_cy").mx_model
live = build()
checks = {}


def both(name, t):
    return getattr(live.Calc, name)(t), getattr(compiled.Calc, name)(t)


def equal(pair):
    a, b = pair
    return float(a) == float(b)


def item(i, t):
    return live.Items[i].v__flat(t), compiled.Items[i].v__flat(t)


old = both("loc_scalar__flat", 1)
checks["before: equal to live"] = equal(old)
ITEM_CELLS = ("pre__rates", "pre_list__rates", "key_local__rates", "key_rebound__rates")


def items_equal(items=(1, 2, 3, 4)):
    return all(float(getattr(live.Items[i], n)(t)) == float(getattr(compiled.Items[i], n)(t))
               for i in items for n in ITEM_CELLS for t in range(4))


checks["item: prefix cell, keys in locals: equal to live"] = items_equal()
# an assigned value of the prefix cell, on an item that has not read it: its reads use the assigned frame
alt = live.Data.rates().loc[("A", "F")] * 7
live.Items[6].sched[()] = alt
compiled.Items[6].sched[()] = alt.copy()
checks["item: prefix cell holding an assigned value: equal to live"] = all(
    float(live.Items[6].pre__rates(t)) == float(compiled.Items[6].pre__rates(t)) for t in range(3))
old_item = item(1, 1)
checks["item: before: equal to live"] = equal(old_item) and all(
    live.Items[1].in__multi(t) == compiled.Items[1].in__multi(t) for t in range(4))
frame2 = live.Data.flat().assign(q=live.Data.flat()["q"] * 2)
live.Data.flat[()] = frame2
compiled.Data.flat[()] = frame2.copy()
# assignment without clear_all: the owner's rule is no automatic clearing on input change, so a result
# cached before stays; a result not cached before reads the NEW frame (no prepared copy survives)
checks["reassign, no clear: Data cell returns the assigned frame"] = compiled.Data.flat().equals(frame2)
checks["reassign, no clear: cached result stays"] = float(compiled.Calc.loc_scalar__flat(1)) == float(old[1])
checks["reassign, no clear: uncached result reads the new frame"] = equal(both("loc_scalar__flat", 2))
checks["item: reassign, no clear: cached result stays"] = float(compiled.Items[1].v__flat(1)) == float(old_item[1])
# a new item binds the frame assigned now; an item that already holds keeps its frame until its Space is
# cleared, as its cached values do (the hold is taken once per item)
checks["item: reassign, no clear: a new item reads the new frame"] = equal(item(3, 2))
checks["item: reassign, no clear: an item that already holds reads the new frame"] = equal(item(1, 2))
compiled.Calc.clear_all()
checks["reassign + Space.clear_all: equal to live on the new frame"] = equal(both("loc_scalar__flat", 1))
compiled.Items.clear_all()
checks["item: reassign + Space.clear_all: equal to live on the new frame"] = equal(item(1, 1)) and equal(item(1, 2))
# Model.clear_all discards assigned inputs on live modelx (the cell recomputes its formula): compiled the same
live.clear_all()
compiled.clear_all()
checks["reassign + Model.clear_all: assignment discarded, equal to live"] = equal(both("loc_scalar__flat", 1)) and \\
    compiled.Data.flat().equals(live.Data.flat())
frame3 = live.Data.flat().assign(q=live.Data.flat()["q"] * 5)
live.Data.flat[()] = frame3
compiled.Data.flat[()] = frame3.copy()
compiled.Calc.clear_all()
live.Calc.clear_all()
checks["second reassign + Space.clear_all: equal to live"] = equal(both("loc_scalar__flat", 3))
# an in-place edit of the bound frame followed by Space.clear_all: as live, the edit is used
live.Data.flat().loc[21, "q"] = 0.5
compiled.Data.flat().loc[21, "q"] = 0.5
compiled.Calc.clear_all()
live.Calc.clear_all()
checks["in-place edit + Space.clear_all: equal to live"] = equal(both("loc_scalar__flat", 1))
compiled.Items.clear_all()
live.Items.clear_all()
checks["item: in-place edit + Space.clear_all: equal to live"] = equal(item(1, 1))
# reassign the table under the prefix cell and rerun: as live after Space.clear_all
rates2 = live.Data.rates() * 3
live.Data.rates[()] = rates2
compiled.Data.rates[()] = rates2.copy()
live.Items.clear_all()
compiled.Items.clear_all()
checks["item: reassign the prefix cell's table + Space.clear_all: equal to live"] = items_equal()


# T-4 item 0: an assigned frame of other kinds than the traced ones (the reads keep pandas), compiled = live
def outcomes(space, cell, ts):
    found = []
    for t in ts:
        try:
            found.append(("ok", float(getattr(space, cell)(t))))
        except Exception as error:
            # live wraps a formula's error in FormulaError and names the original in its message (as checks.py
            # reads it): compare the original exception type
            name = type(error).__name__
            if name == "FormulaError":
                named = re.search(r"\\n(\\w+): ", str(error))
                name = named.group(1) if named else name
            found.append(("raise", name))
    return found


def same_outcomes(cell, ts, items=False):
    if items:
        return all(outcomes(live.Items[i], cell, ts) == outcomes(compiled.Items[i], cell, ts) for i in (1, 2))
    return outcomes(live.Calc, cell, ts) == outcomes(compiled.Calc, cell, ts)


base = live.Data.flat()
KINDS = {
    "int level -> float level": base.set_axis(pd.Index([20.0, 21.0, 22.0, 23.0], name="age"), axis=0),
    "float column -> int column": base.assign(q=[1, 2, 3, 4]),
    "int level -> text level": base.set_axis(pd.Index(["20", "21", "22", "23"], name="age"), axis=0),
}
for name, frame in KINDS.items():
    live.Data.flat[()] = frame
    compiled.Data.flat[()] = frame.copy()
    live.Calc.clear_all(); compiled.Calc.clear_all(); live.Items.clear_all(); compiled.Items.clear_all()
    checks[f"kind: {name}: equal to live"] = (same_outcomes("loc_scalar__flat", range(4))
                                              and same_outcomes("at__flat", range(4))
                                              and same_outcomes("v__flat", range(4), items=True))
multi = live.Data.multi()
multi_int = multi.set_axis(multi.index.set_levels([1, 2, 3], level=0), axis=0)
live.Data.multi[()] = multi_int
compiled.Data.multi[()] = multi_int.copy()
live.Calc.clear_all(); compiled.Calc.clear_all(); live.Items.clear_all(); compiled.Items.clear_all()
checks["kind: text level -> int level (two levels): equal to live"] = (
    same_outcomes("loc_tuple__multi", range(4)) and same_outcomes("at_tuple__multi", range(4)))
json.dump(checks, sys.stdout, indent=1)
'''

# Contract checks that fail, removed step by step. Empty since T-3; at 84befe4 the uncached call after a
# reassignment read the old frame.
KNOWN_CONTRACT = set()

# Documented differences from live modelx (owner's decision), each with the sentence a user needs
DOCUMENTED_CONTRACT = {
    'item: reassign, no clear: an item that already holds reads the new frame':
        'An item keeps the frame it bound until its Space is cleared, like its cached values: apply an input '
        'change by reassigning it and clearing the dependent Space (Space.clear_all()).',
}


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, *args], cwd=tmp_path, check=True, timeout=2400,
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    work = tmp_path_factory.mktemp("probe")
    (work / "probe_model.py").write_text(MODEL.format(ops=OPS, stmts=STMTS, data=DATA, t5_items=T5_ITEMS),
                                         encoding="utf-8")
    (work / "sample.py").write_text(SAMPLE, encoding="utf-8")
    (work / "checks.py").write_text(CHECKS, encoding="utf-8")
    (work / "contract.py").write_text(CONTRACT, encoding="utf-8")
    (work / "spec.py").write_text(repr({"parallel_workers": 1}), encoding="utf-8")
    _run(work, "probe_model.py")
    build = _run(work, "-m", "modelx_cython", "probe_nomx", "--sample", "sample.py", "--spec", "spec.py")
    verdicts = json.loads(_run(work, "checks.py").stdout)
    return work, build.stdout + build.stderr, verdicts


def test_probe_compiled_equals_live(probe):
    _, _, verdicts = probe
    differing = {name: v for name, v in verdicts.items()
                 if v != "same" and not (v.startswith("boundary: ")
                                         and set(v[len("boundary: "):].split("; ")) <= BOUNDARY)}
    unexpected = {name: v for name, v in differing.items() if name not in KNOWN}
    fixed = sorted(name for name in KNOWN if name not in differing)
    assert not unexpected, "compiled differs from live where it did not before:\n" + "\n".join(
        f"{n}: {v}" for n, v in sorted(unexpected.items()))
    assert not fixed, f"now equal to live, remove from KNOWN: {fixed}"


def test_reassign_and_clear(probe):
    work, _, _ = probe
    checks = json.loads(_run(work, "contract.py").stdout)
    failing = {name for name, ok in checks.items() if not ok}
    expected = KNOWN_CONTRACT | set(DOCUMENTED_CONTRACT)
    assert failing == expected, f"failing: {sorted(failing)}; expected failing: {sorted(expected)}"


def test_item_holds_take_no_lookup_or_lock(probe):
    """T-4.2: an item binds each shared Data table with one read of the table the Data space published
    (``_mx_prepared_<name>()``): its per-item hold code calls no cache lookup (prepared/bind_ref), no assigned-value
    lookup and takes no lock. A frame the item computes itself (a cell of its own, a new frame per item) is bound
    by preparing it. Read from the generated module of the probe."""
    work, _, _ = probe
    source = (work / "probe_nomx_cy" / "_mx_classes.py").read_text(encoding="utf-8")
    start = source.index("class _c_Items(")
    end = source.find("\nclass ", start + 1)
    items = source[start:end if end > 0 else len(source)]
    # the item's formulas live in module-level dense helpers too: those taking an Items receiver
    helpers = [block for block in source.split("\n@_mx_cy.cfunc") if "_mx_private_receiver[_c_Items]" in block]
    code = items + "".join(helpers)
    bindings = [line.strip() for line in code.splitlines() if re.match(r"\s*self\._mx_tp_\d+ = (?!None)", line)]
    assert bindings, "no item binding found"
    shared = [line for line in bindings if "self.data." in line]
    assert shared and all("._mx_prepared_" in line for line in shared), shared
    for forbidden in ("bind_ref(", ".prepared(", "assigned(", "hold_unassigned(", "_mx_lock"):
        assert not [line for line in code.splitlines()
                    if forbidden in line and "_mx_tp_" in line and "self.data." in line], forbidden
    # T-5: every read kind (keyed reads, views of a prefix, index walks, row locals) uses a held table only borrowed,
    # as a PreparedRef (no reference counted per step); the table object itself appears only in the hold statements
    # (once per item: the binding, the held codes and rows, the resets)
    borrowed = re.compile(r"_mx_cy\.cast\(_mx_helper_\w+\.PreparedRef, self\._mx_tp_\d+\)")
    hold_line = re.compile(r"\s*(if )?(self|root)\._mx_t[pqkw]_\d+ (=|==) ")
    for kind in T5_ITEMS:
        assert any(kind in block for block in helpers) or kind in items, kind
    identity = re.compile(r"self\._mx_tp_\d+ is self\._mx_tp_\d+")      # a pointer comparison
    escaping = [line.strip()[:160] for line in code.splitlines()
                if "_mx_tp_" in identity.sub("", borrowed.sub("", line)) and not hold_line.match(line)
                and not re.match(r"\s*_mx_tp_\d+: object", line)]
    assert not escaping, escaping


def test_native_branches_call_no_pandas(probe):
    """T-5: the native branch of every lowered read (the body of its ``... if <slot> > 0 ... else <pandas>``), with
    the pandas branches of the reads nested in it removed, makes no pandas call (a read nested in a key part is
    lowered there too)."""
    import libcst as cst
    work, _, _ = probe
    module = cst.parse_module((work / "probe_nomx_cy" / "_mx_classes.py").read_text(encoding="utf-8"))
    code = module.code_for_node
    found = []

    class Strip(cst.CSTTransformer):
        def leave_IfExp(self, original, updated):
            return updated.with_changes(orelse=cst.Name("None")) if ".slot(" in code(original.test) else updated

    class Scan(cst.CSTVisitor):
        def visit_IfExp(self, node):
            if ".slot(" in code(node.test):
                native = re.sub(r"'[^']*'|\"[^\"]*\"", "''", code(node.body.visit(Strip())))   # not in a literal
                if any(token in native for token in (".loc[", ".at[", ".iloc", ".iat[", ".index", "self.data.")):
                    found.append(native[:200])
    module.visit(Scan())
    assert not found, found

