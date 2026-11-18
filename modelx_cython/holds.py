"""What an item class holds from shared inputs: one registry per class (R-4).

Compiled per-item code holds what it reads from a shared input. The first read in an item's lifetime
resolves the value into a field of the item, and later reads use that field borrowed. The fields are
reset with the item's per-policy caches.

A class's ``HoldTable`` has one entry per held value, in three kinds:

- ``handle``: the TableHandle or RecordHandle of a zero-argument cached table or record Cell
  (``_mx_th_<n>``; None until resolved), keyed by the source expression of the Cell call;
- ``column``: the column index, or record field slot, of one literal name read through a held handle
  (``_mx_tc_<n>``; 0 until resolved), keyed by (held field, name);
- ``row``: the row of a key fixed per item and read through a held handle (``_mx_tr_<n>``; 0 until
  resolved), keyed by (held field, key parts);
- ``prepared`` and ``prepared_slot`` (T-3): the PreparedTable of a pandas frame and the slot of a column in it.

Which reads are held is decided in ``table_reads`` by traced type and data flow, never by how a read is
spelled. This module only names the fields and gives their declarations and reset lines.
"""

_KINDS = {
    # kind: (field prefix, C type of the .pxd, annotation of the .py, reset value)
    'handle': ('_mx_th_', 'object', 'object', 'None'),
    'column': ('_mx_tc_', 'Py_ssize_t', '_mx_cy.Py_ssize_t', '0'),
    'row': ('_mx_tr_', 'Py_ssize_t', '_mx_cy.Py_ssize_t', '0'),
    # T-3 (frame_reads): the PreparedTable of a pandas frame read per item, and the slot of one column in it
    # (0 until resolved, as a new item's field starts; -1 when the read stays pandas; slot + 1 when native),
    # keyed by (receiver, column, kind)
    'prepared': ('_mx_tp_', 'object', 'object', 'None'),
    'prepared_slot': ('_mx_tq_', 'Py_ssize_t', '_mx_cy.Py_ssize_t', '0'),
    # T-4: one plus the level code of a key part fixed per item (-1 absent), keyed by (table field, level, part);
    # one plus the row of a row cell's key (-1: pandas), keyed by (table field, key)
    'prepared_code': ('_mx_tk_', 'Py_ssize_t', '_mx_cy.Py_ssize_t', '0'),
    'prepared_row': ('_mx_tw_', 'Py_ssize_t', '_mx_cy.Py_ssize_t', '0'),
}


class HoldTable:
    """The held fields of one item class, by kind and key, in the order they were first needed."""

    def __init__(self):
        self._fields = {kind: {} for kind in _KINDS}

    def field(self, kind, key):
        """The field holding ``key`` (created on first use, numbered per kind)."""
        fields = self._fields[kind]
        field = fields.get(key)
        if field is None:
            field = fields[key] = f'{_KINDS[kind][0]}{len(fields)}'
        return field

    def fields(self):
        """(field, kind) of every held field: handles, then columns, then rows."""
        for kind, fields in self._fields.items():
            for field in fields.values():
                yield field, kind

    def __bool__(self):
        return any(self._fields.values())


def hold_table(cls_info):
    """The class's HoldTable, created on first use."""
    table = getattr(cls_info, 'holds', None)
    if table is None:
        table = cls_info.holds = HoldTable()
    return table


def _held(cls_info):
    table = getattr(cls_info, 'holds', None)
    return table.fields() if table is not None else ()


def pxd_declarations(cls_info):
    """``cdef`` lines of the class's .pxd for its held fields."""
    return [f'cdef {_KINDS[kind][1]} {field}\n' for field, kind in _held(cls_info)]


def py_declarations(cls_info):
    """Annotated declarations of the class body of the .py for its held fields."""
    return [f'{field}: {_KINDS[kind][2]}\n' for field, kind in _held(cls_info)]


def reset_lines(cls_info, target='root'):
    """Reset of every held field of ``target``, with the per-policy caches."""
    return [f'{target}.{field} = {_KINDS[kind][3]}' for field, kind in _held(cls_info)]
