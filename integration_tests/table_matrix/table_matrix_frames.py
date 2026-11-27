"""Pandas frames, partial selections, and label cases for the read matrix."""

import enum
import warnings

import numpy as np
import pandas as pd


Lbl = enum.IntEnum('Lbl', {('M%d' % -v if v < 0 else 'V%d' % v): v for v in range(-5, 41)})


def _frames():
    pmi = pd.MultiIndex
    return {
        'C': pd.DataFrame({'q': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]}, index=pd.Index(range(-3, 4))),
        'R2': pd.DataFrame({'q': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
                           index=pmi.from_product([[2, 1], [10, 20, 30]])),
        'R': pd.DataFrame({'q': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
                          index=pmi.from_product([[2, 1], ['b', 'a', 'c']])),
        'R3': pd.DataFrame({'q': np.arange(12.0) / 8},
                           index=pmi.from_product([[1, 0], ['y', 'x'], [5, 6, 7]])),
        'F1': pd.DataFrame({'q': [0.1, 0.3, 0.7, 2.0], 'b': [1.0, 3.0, 7.0, 20.0]},
                           index=pd.Index([1, 3, 7, 20])),
        'F2': pd.DataFrame({'q': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'b': [0.5, 0.25, 0.125, 1.5, 2.5, 3.5]},
                           index=pmi.from_tuples([(0, 'x', 1), (0, 'x', 2), (0, 'y', 1), (1, 'x', 3),
                                                  (1, 'y', 1), (1, 'y', 2)])),
        'F2B': pd.DataFrame({'q': [1.0, 2.0, 3.0, 4.0]},
                            index=pmi.from_tuples([(True, 'x', 1), (True, 'y', 2), (False, 'x', 1),
                                                   (False, 'x', 3)])),
        'G': pd.DataFrame({'q': [0.25, 0.5, 0.75], 'n': [3, 2, 1], 'flag': [True, False, True],
                           's': ['MALE', 'FEMALE', 'MALE'], 'e': [Lbl.V0, Lbl.V1, Lbl.V0]},
                          index=pd.Index([10, 20, 30])),
        'GS': pd.DataFrame({'q': [1.25, 2.5, 3.75], 'n': [7, 8, 9]}, index=pd.Index(['a', 'b', 'c'])),
        'O_C': pd.DataFrame({'q': [0.125]}, index=pd.Index([5])),
        'O_R': pd.DataFrame({'q': [9.5]}, index=pmi.from_product([[7], ['x']])),
        'O_G': pd.DataFrame({'q': [1.25], 'n': [4], 's': ['A']}, index=pd.Index([2])),
    }


FRAMES = _frames()
# variant -> (root frame, prefix or None)
LAYOUT = {name: (name, None) for name in FRAMES}
LAYOUT.update({'S_R3': ('R3', (1,)), 'S_R3C': ('R3', (0, 'x')), 'S_RF': ('R', (2,)),
               'S_F2': ('F2', (1,)), 'S_F2F': ('F2', (0, 'x')), 'S_R2': ('R2', (1,)),
               'S_F2B': ('F2B', (True,))})
VARIANTS = {'C': ['C'], 'R': ['R2', 'R', 'R3'], 'F1': ['F1'], 'F2': ['F2', 'F2B'], 'G': ['G', 'GS'],
            'S': ['S_R3', 'S_R3C', 'S_RF', 'S_F2', 'S_F2F', 'S_R2', 'S_F2B'],
            'O': ['O_C', 'O_R', 'O_G']}


def frame_of(variant):
    root, prefix = LAYOUT[variant]
    frame = FRAMES[root]
    if prefix is None:
        return frame
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return frame.loc[prefix if len(prefix) > 1 else prefix[0]]


def levels(variant):
    return frame_of(variant).index.nlevels


# ---------------------------------------------------------------- keys by key type

def _cast(part, kt):
    """An int label part as key type kt (None: not representable); other parts unchanged."""
    if type(part) is not int:
        return part
    if kt == 'enum':
        return Lbl(part)
    if kt == 'np.int64':
        return np.int64(part)
    if kt == 'np.uint64':
        return np.uint64(part) if part >= 0 else None
    if kt == 'np.float32':
        return np.float32(part)
    if kt == 'np.float64':
        return np.float64(part)
    if kt == 'float':
        return float(part)
    return part


def _natural(variant):
    """(hit keys, miss keys) as tuples of label parts."""
    frame = frame_of(variant)
    labels = [key if type(key) is tuple else (key,) for key in frame.index]
    labels = [tuple(part.item() if isinstance(part, np.generic) else part for part in key)
              for key in labels]
    hits = list(dict.fromkeys([labels[0], labels[len(labels) // 2], labels[-1]]))
    misses = []
    for j in range(len(labels[0])):
        parts = [key[j] for key in labels]
        if all(type(part) is int for part in parts):
            for value in (max(parts) + 1, min(parts) - 1):
                misses.append(labels[0][:j] + (value,) + labels[0][j + 1:])
        elif all(type(part) is str for part in parts):
            misses.append(labels[0][:j] + ('zz',) + labels[0][j + 1:])
        elif all(type(part) is bool for part in parts) and len(set(parts)) == 1:
            misses.append(labels[0][:j] + (not parts[0],) + labels[0][j + 1:])
    return hits, [key for key in misses if key not in labels]


def keys(variant, kt):
    """(hits, misses) for key type kt: each a tuple of key parts as passed."""
    hits, misses = _natural(variant)
    if kt == 'str':
        # the str parts decide: hits are natural keys, misses change a str part (or are str
        # keys on a level that holds no str)
        str_levels = [j for j, part in enumerate(hits[0]) if type(part) is str]
        if str_levels:
            j = str_levels[0]
            return hits, [hits[0][:j] + ('zz',) + hits[0][j + 1:]]
        return [], [('zz',) + hits[0][1:], ('1',) + hits[0][1:]]
    if kt == 'bool':
        bool_levels = [j for j, part in enumerate(hits[0]) if type(part) is bool]
        if bool_levels:
            return hits, misses
        return [], [(True,) + hits[0][1:], (False,) + hits[0][1:]]
    if kt == 'tuple':
        return hits, misses
    cast = lambda key: tuple(_cast(part, kt) for part in key)  # noqa: E731
    hits = [cast(key) for key in hits]
    misses = [cast(key) for key in misses]
    if kt in ('float', 'np.float32', 'np.float64'):
        number = [j for j, part in enumerate(hits[0]) if isinstance(part, (float, np.floating))]
        if number:
            j = number[0]
            misses.append(hits[0][:j] + (_cast_float(kt, 1.5),) + hits[0][j + 1:])
    return ([key for key in hits if None not in key], [key for key in misses if None not in key])


def _cast_float(kt, value):
    return {'float': float, 'np.float32': np.float32, 'np.float64': np.float64}[kt](value)


def label(parts):
    return parts[0] if len(parts) == 1 else parts


def reads(form, operation, key_type):
    """Dynamic label cases in addition to the formulas with a natural key type."""
    for variant in VARIANTS[form]:
        hits, misses = keys(variant, key_type)
        depths = range(1, levels(variant))
        if operation == 1:
            for key in hits + misses:
                yield 'loc', variant, (label(key), 'q')
        elif operation == 4:
            for key in hits + misses:
                yield 'has', variant, (label(key),)
                for depth in depths:
                    yield 'has', variant, (label(key[:depth]),)
        elif operation == 6:
            for key in hits + misses:
                for depth in depths:
                    prefix, suffix = label(key[:depth]), label(key[depth:])
                    yield 'sub_loc', variant, (prefix, suffix, 'q')
                    yield 'sub_has', variant, (prefix, suffix)
                    yield 'sub_cell', variant, (prefix, suffix, 'q')
        elif operation == 10:
            for key in misses:
                yield 'loc', variant, (label(key), 'q')
                yield 'at', variant, (label(key), 'q')
            if hits:
                yield 'loc', variant, (label(hits[0]), 'zz')
                yield 'column', variant, ('zz', label(hits[0]))
        else:
            raise ValueError(f'No dynamic label cases for operation {operation}')


def numeric_boundaries():
    first = int(FRAMES['C'].index.min())
    for key in (2**63 - 1, -2**63, 2**63, -2**63 - 1, 2**64, first - 1, first - 2**40):
        yield 'loc', 'C', (key, 'q')
        yield 'has', 'C', (key,)
