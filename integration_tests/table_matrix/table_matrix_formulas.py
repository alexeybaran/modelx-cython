"""Model Cell spellings for the representative cases of the pandas read matrix."""

import enum

import pandas as pd


class Sex(enum.IntEnum):
    M = 0
    F = 1


def frames():
    f2 = pd.DataFrame({'sex': [Sex.M, Sex.M, Sex.F, Sex.F] * 2, 'plan': ['A', 'B'] * 4,
                       'year': [1, 1, 1, 1, 2, 2, 2, 2], 'v': [float(i) for i in range(8)],
                       'w': [0.5 * i for i in range(8)]}).set_index(['sex', 'plan', 'year'])
    return {
        'C': pd.DataFrame({'q': [0.25, 0.5, 0.75, 1.0]}, index=pd.Index([5, 6, 7, 8], name='k')),
        'R': pd.DataFrame({'q': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
                          index=pd.MultiIndex.from_product([[1, 2], [10, 20, 30]], names=['a', 'b'])),
        'F1': pd.DataFrame({'v': [0.5, 1.5, 2.5, 3.5], 'w': [9.0, 8.0, 7.0, 6.0]},
                           index=pd.Index([1, 2, 5, 9], name='k')),
        'F2': f2,
        'G': pd.DataFrame({'f': [1.5, 2.5, 3.5], 'n': [10, 20, 30], 'b': [True, False, True],
                           's': ['x', 'y', 'z']}, index=pd.Index([1, 2, 3], name='k')),
        'O': pd.DataFrame({'q': [4.5]}, index=pd.Index([7], name='k')),
    }


# Cells returning a prefix selection or a model-point row.
DERIVED = {
    'S_F2': 'def S_F2():\n    return T_F2().loc[(Sex.F, "B")]',
    'S_R': 'def S_R():\n    return T_R().loc[2]',
    'R_G': 'def R_G():\n    return T_G().loc[2]',
}

F1KEY = '(1, 2, 5, 9)[i % 4]'
# (form, operation, key) -> (formula body using argument i, arguments)
SQUARES = {
    ('C', 1, 'int'): ('return float(T_C().loc[5 + i % 4, "q"])', range(4)),
    ('C', 2, 'int'): ('return T_C().at[5 + i % 4, "q"]', range(4)),
    ('C', 4, 'int'): ('return (4 + i) in T_C().index', range(7)),
    ('C', 5, 'int'): ('return float(T_C().loc[min(max(3 + i, int(T_C().index.min())), int(T_C().index.max())), "q"])',
                      range(9)),
    ('C', 7, 'int'): ('row = T_C().loc[5 + i % 4]\n    x = row["q"]\n    row = T_C().loc[5]\n    return x + row["q"]',
                      range(4)),
    ('C', 8, 'int'): ('sub = T_C()\n    return float(sub.loc[sub.index.asof(5 + i), "q"]) + sub.index.asof(5 + i)',
                      range(6)),
    ('C', 9, 'int'): ('sub = T_C()\n    return sub.index[sub.index.searchsorted(4 + i % 4, side="right")]'
                      ' + sub.index[i % 4]', range(4)),
    ('C', 10, 'int'): ('return T_C().loc[9, "q"] if i == 0 else T_C().loc[5, "nope"] if i == 1 else T_C().loc[4, "q"]',
                       range(3)),
    ('C', 12, 'int'): ('return T_C()["q"][5 + i % 4]', range(4)),
    ('R', 1, 'int'): ('return float(T_R().loc[(1 + i % 2, 10 * (1 + i % 3)), "q"])', range(6)),
    ('R', 2, 'int'): ('return T_R().at[(1 + i % 2, 10 * (1 + i % 3)), "q"]', range(6)),
    ('R', 4, 'int'): ('return ((1 + i % 3, 10 * (1 + i % 4)) in T_R().index, (1 + i % 3) in T_R().index)',
                      range(12)),       # full key, and a one-part prefix of two levels (D27a, D54)
    ('R', 5, 'int'): ('return T_R().index.max(), T_R().index.min()', range(1)),
    ('R', 6, 'int'): ('sub = T_R().loc[1 + i % 2]\n    return float(sub.loc[10 * (1 + i % 3), "q"]) + sub.index.max()',
                      range(6)),
    ('R', 7, 'int'): ('row = T_R().loc[(1 + i % 2, 20)]\n    return row["q"]', range(2)),
    ('R', 10, 'int'): ('return T_R().loc[(1, 10), "q"] if i == 0 else T_R().loc[(3, 10), "q"] if i == 1 '
                       'else T_R().loc[(1, 40), "q"]', range(3)),
    ('R', 12, 'int'): ('return T_R()["q"][(1 + i % 2, 30)]', range(2)),
    ('F1', 1, 'int'): (f'return float(T_F1().loc[{F1KEY}, "v"])', range(4)),
    ('F1', 2, 'int'): (f'return T_F1().at[{F1KEY}, "w"]', range(4)),
    ('F1', 4, 'int'): ('return i in T_F1().index', range(11)),
    ('F1', 5, 'int'): ('tbl = T_F1()\n    return float(tbl.loc[min(max(i, int(tbl.index.min())), int(tbl.index.max())), "v"])',
                       range(12)),
    ('F1', 7, 'int'): (f'row = T_F1().loc[{F1KEY}]\n    x = row["v"]\n    if i > 1:\n        row = T_F1().loc[1]\n'
                       '    return x + row["w"]', range(4)),
    ('F1', 8, 'int'): ('sub = T_F1()\n    return float(sub.loc[sub.index.asof(1 + i), "v"]) + 2 * sub.index.asof(1 + i)',
                       range(10)),
    ('F1', 9, 'int'): ('sub = T_F1()\n    return sub.index[sub.index.searchsorted(i, side="right")] + sub.index[-1 - i % 4]'
                       ' + sub.index.searchsorted(i)', range(9)),
    ('F1', 10, 'int'): ('sub = T_F1()\n    return sub.loc[3, "v"] if i == 0 else sub.loc[1, "nope"] if i == 1 else '
                        'sub.index.asof(0)', range(3)),
    ('F1', 12, 'int'): (f'return T_F1()["v"][{F1KEY}]', range(4)),
    ('F2', 1, 'tuple'): ('return float(T_F2().loc[(Sex(i % 2), "AB"[i % 2], 1 + i // 2 % 2), "v"])', range(4)),
    ('F2', 2, 'tuple'): ('return T_F2().at[(Sex(i % 2), "AB"[i % 2], 1 + i // 2 % 2), "w"]', range(4)),
    ('F2', 4, 'tuple'): ('return ((Sex(i % 2), "ABC"[i % 3]) in T_F2().index, (Sex.F, "B", 1 + i % 3) in T_F2().index)',
                         range(6)),
    ('F2', 5, 'tuple'): ('return T_F2().index.max(), T_F2().index.min()', range(1)),
    ('F2', 6, 'tuple'): ('sub = T_F2().loc[(Sex(i % 2), "AB"[i // 2 % 2])]\n'
                         '    return float(sub.loc[1, "v"]) + float(sub.loc[2, "w"]) + sub.index.max()', range(4)),
    ('F2', 7, 'tuple'): ('row = T_F2().loc[(Sex(i % 2), "AB"[i % 2], 2)]\n    return row["v"] + row["w"]', range(2)),
    ('F2', 10, 'tuple'): ('return T_F2().loc[(Sex.M, "C", 1), "v"] if i == 0 else T_F2().loc[(Sex.M, "A", 1), "nope"]',
                          range(2)),
    ('F2', 12, 'tuple'): ('return T_F2()["v"][(Sex.F, "A", 2)]', range(1)),
    ('G', 1, 'int'): ('return float(T_G().loc[1 + i % 3, "f"])', range(3)),
    ('G', 2, 'int'): ('return T_G().at[1 + i % 3, "s"]', range(3)),
    ('G', 3, 'int'): ('return int(T_G().loc[1 + i % 3, "n"]) + int(T_G().loc[1 + i % 3, "b"])', range(3)),
    ('G', 4, 'int'): ('return i in T_G().index', range(5)),
    ('G', 5, 'int'): ('return int(T_G().index.min()) + int(T_G().index.max())', range(1)),
    ('G', 7, 'int'): ('row = T_G().loc[1 + i % 3]\n    return (row["f"], row["n"], row["b"], row["s"])', range(3)),
    ('G', 8, 'int'): ('sub = T_G()\n    return sub.loc[sub.index.asof(1 + i), "s"]', range(5)),
    ('G', 9, 'int'): ('sub = T_G()\n    return sub.index[sub.index.searchsorted(i % 3, side="right")]', range(3)),
    ('G', 10, 'int'): ('return T_G().loc[4, "f"] if i == 0 else T_G().loc[1, "nope"]', range(2)),
    ('G', 11, 'str'): ('rec = R_G()\n    return (R_G()["f"] * 2 + R_G()["n"], rec["s"], float(rec["b"]))', range(1)),
    ('G', 12, 'int'): ('return T_G()["n"][1 + i % 3]', range(3)),
    ('S', 1, 'int'): ('return float(S_F2().loc[1 + i % 2, "v"])', range(2)),
    ('S', 2, 'int'): ('return S_F2().at[1 + i % 2, "w"]', range(2)),
    ('S', 4, 'int'): ('return i in S_F2().index', range(4)),
    ('S', 5, 'int'): ('return S_F2().index.max() - S_F2().index.min()', range(1)),
    ('S', 7, 'int'): ('row = S_F2().loc[1 + i % 2]\n    return row["v"]', range(2)),
    ('S', 8, 'int'): ('sub = S_F2()\n    return float(sub.loc[sub.index.asof(1 + i), "v"])', range(4)),
    ('S', 9, 'int'): ('sub = S_R()\n    return sub.index[sub.index.searchsorted(10 * i, side="right")]', range(3)),
    ('S', 10, 'int'): ('return S_F2().loc[3, "v"]', range(1)),
    ('S', 12, 'int'): ('return S_R()["q"][10 * (1 + i % 3)]', range(3)),
    ('O', 1, 'int'): ('return float(T_O().loc[7, "q"])', range(1)),
    ('O', 2, 'int'): ('return T_O().at[7, "q"]', range(1)),
    ('O', 4, 'int'): ('return (6 + i) in T_O().index', range(3)),
    ('O', 5, 'int'): ('return T_O().index.min() + T_O().index.max()', range(1)),
    ('O', 7, 'int'): ('row = T_O().loc[7]\n    return row["q"]', range(1)),
    ('O', 8, 'int'): ('sub = T_O()\n    return float(sub.loc[sub.index.asof(7 + i), "q"])', range(3)),
    ('O', 9, 'int'): ('sub = T_O()\n    return sub.index.searchsorted(6 + i, side="right")', range(3)),
    ('O', 10, 'int'): ('return T_O().loc[8, "q"] if i == 0 else T_O().loc[7, "nope"]', range(2)),
    ('O', 12, 'int'): ('return T_O()["q"][7]', range(1)),
}
