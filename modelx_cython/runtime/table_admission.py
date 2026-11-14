"""The one conversion of a plain pandas frame into the internal table (auto-tables, D104).

``admit(value, cell, nargs=0)`` is used at build time, on the value a zero-argument cell returned
in the sample (``modelx_cython.auto_tables``), and at run time, on a compiled table cell's formula
result and on an assigned DataFrame (``_native_inputs.table_from_value``, api-tables D101). The
decision is made on the value only, never on how the formula is written.

A pandas DataFrame is admitted when every condition holds:

  (C1, "the cell has no arguments", was dropped by D105: which cells are admitted is decided at build
  by the native reads that consume them, whatever their arity, D106)
  C2  the value is a DataFrame (a Series is not admitted: it stays pandas)
  C3  its keys (the index) are unique
  C4  no key is missing (NaN, None, NaT)
  C5  every key level holds one kind of label: int, float, str, bool, or members of one enum class
  C6  the column names are non-empty unique strings
  C7  every column is numeric, bool, object or string dtype
  C8  ``modelx_helpers.pdn.table`` accepts the frame (its own scalar-kind contract)

The result is the helper Table ``pdn.table`` builds, given the frame's pandas vocabulary
(``immutable_inputs.with_frame``): an operation the Table does not answer runs pandas on one
read-only copy of the frame, which is also the DataFrame Python callers of a compiled model get.
An unnamed RangeIndex is admitted (row numbers are the keys). A helper Table, or a value that is not
pandas, is returned unchanged, so ``pdn.table(...)`` in model text keeps working.

``TableAdmissionError`` (a ValueError) lists every failed condition, one line each:
``<cell>: <reason>``.
"""

import numbers


class TableAdmissionError(ValueError):
    """A pandas value that cannot become the internal table; ``reasons`` lists the failed conditions."""

    def __init__(self, cell, reasons):
        self.cell, self.reasons = cell, tuple(reasons)
        super().__init__('\n'.join(f'{cell}: {reason}' for reason in self.reasons))


def _label_kind(label):
    if isinstance(label, bool):
        return 'bool'
    if isinstance(label, numbers.Integral):
        cls = type(label)
        return f'enum {cls.__qualname__}' if hasattr(cls, '__members__') else 'int'
    if isinstance(label, numbers.Real):
        return 'float'
    if isinstance(label, str):
        cls = type(label)
        return f'enum {cls.__qualname__}' if hasattr(cls, '__members__') else 'str'
    if hasattr(type(label), '__members__'):
        return f'enum {type(label).__qualname__}'
    return type(label).__name__


def _is_missing(label):
    try:
        return label is None or label != label
    except Exception:       # noqa: BLE001 - an exotic label that cannot compare is not missing
        return False


def conditions(value, nargs=0):
    """The reasons (C2-C8) why ``value`` cannot be admitted, in condition order; [] if none.

    ``nargs`` (the cell's argument count) does not decide (D105). C8 is checked by :func:`admit`,
    which builds the table.
    """
    import pandas as pd
    reasons = []
    if isinstance(value, pd.Series):
        reasons.append('returns a pandas Series: not admitted as a table (C2); return a keyed DataFrame '
                       '(e.g. series.to_frame()) to have it admitted, or a scalar')
        return reasons
    if not isinstance(value, pd.DataFrame):
        reasons.append(f'returns {type(value).__name__}, not a pandas DataFrame (C2)')
        return reasons
    index = value.index
    names = list(index.names)
    if index.has_duplicates:
        duplicated = index[index.duplicated()]
        reasons.append(f'duplicate key {duplicated.tolist()[0]!r} ({len(duplicated)} duplicated rows) (C3); '
                       'keys must be unique')
    for level in range(index.nlevels):
        values = index.get_level_values(level)
        name = names[level]
        if values.hasnans:
            reasons.append(f'missing value in key level {name!r} (C4)')
            values = values.dropna()
        kind = getattr(values.dtype, 'kind', None)
        if kind in ('i', 'u', 'f', 'b'):       # a numpy level holds one kind of label
            kinds = [{'i': 'int', 'u': 'int', 'f': 'float', 'b': 'bool'}[kind]] if len(values) else []
        else:                                   # object or string levels: each distinct label
            labels = [label for label in values.unique().tolist() if not _is_missing(label)]
            kinds = sorted({_label_kind(label) for label in labels})
        if len(kinds) > 1:
            reasons.append(f'key level {name!r} mixes {", ".join(kinds)} labels (C5); one kind per level')
        elif kinds and kinds[0] not in ('int', 'float', 'str', 'bool') and not kinds[0].startswith('enum '):
            reasons.append(f'key level {name!r} holds {kinds[0]} labels (C5); keys are int, float, str, bool or '
                           'enum members')
    columns = list(value.columns)
    if (not columns or any(type(column) is not str or not column for column in columns)
            or len(set(columns)) != len(columns)):
        reasons.append(f'column names must be non-empty unique strings (C6), got {columns[:6]!r}'
                       + (' ...' if len(columns) > 6 else ''))
    else:
        for column, dtype in zip(columns, value.dtypes):
            if getattr(dtype, 'kind', None) in tuple('fiubO') or isinstance(dtype, pd.StringDtype):
                continue
            reasons.append(f'column {column!r} has unsupported dtype {dtype} (C7); numeric, bool, object or '
                           'string columns are admitted')
    return reasons


def display_name(cell):
    """``Outer.Space.cell`` for a generated name such as ``pkg._m_Outer._mx_classes._c_Space.cell``."""
    if '._c_' not in cell:
        return cell
    parts = cell.split('.')
    owners = [part[3:] for part in parts if part.startswith('_m_')]
    at = max(i for i, part in enumerate(parts) if part.startswith('_c_'))
    rest = [part[3:] if part.startswith('_f_') else part for part in parts[at + 1:]]
    return '.'.join((*owners, parts[at][3:], *rest))


def admit(value, cell, nargs=0):
    """The internal table for ``value``, the value of ``cell``; see the module docstring.

    Returns ``value`` itself when it is not a pandas object (a helper Table, a scalar, ...).
    Raises :class:`TableAdmissionError` naming the cell and every failed condition.
    """
    module = type(value).__module__
    if module.split('.')[0] != 'pandas':
        return value
    cell = display_name(cell)
    reasons = conditions(value, nargs)
    if reasons:
        raise TableAdmissionError(cell, reasons)
    from modelx_helpers import pdn
    from modelx_helpers.immutable_inputs import with_frame
    try:
        table = pdn.table(value)
    except (TypeError, ValueError) as error:
        raise TableAdmissionError(cell, [f'{error} (C8)']) from error
    from .table_cells import ReadOnlyFrame
    frame = ReadOnlyFrame._wrap(value.copy(deep=True), table, cell)
    return with_frame(table, frame)
