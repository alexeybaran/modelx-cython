"""What an item class holds from shared inputs: one registry per class.

Compiled per-item code holds what it reads from a shared input. The first read in an item's lifetime
resolves the value into a field of the item, and later reads use that field borrowed. The fields are
reset with the item's per-policy caches.

A class's ``HoldTable`` has one entry per held value. The one kind here is ``array``: the slice of a
zero-argument cached Cell with a typed-memoryview result read by subscript (``_mx_ta_<n>`` of the Cell's
memoryview type, valid while ``_mx_ta_<n>_ready``), keyed by (receiver, Cell); see ``held_arrays``.
This module only names the fields and gives their declarations and reset lines.
"""

_KINDS = {
    # kind: (field prefix, C type of the .pxd, annotation of the .py, reset value)
    'array': ('_mx_ta_', None, None, None),     # typed per field; reset clears the ready flag
}


class HoldTable:
    """The held fields of one item class, by kind and key, in the order they were first needed."""

    def __init__(self):
        self._fields = {kind: {} for kind in _KINDS}
        self.types = {}     # array field -> (C type of the .pxd, annotation of the .py)

    def field(self, kind, key, ctype=None, annotation=None):
        """The field holding ``key`` (created on first use, numbered per kind); an ``array`` field takes the
        memoryview's C type and annotation."""
        fields = self._fields[kind]
        field = fields.get(key)
        if field is None:
            field = fields[key] = f'{_KINDS[kind][0]}{len(fields)}'
            if kind == 'array':
                self.types[field] = (ctype, annotation)
        return field

    def fields(self):
        """(field, kind) of every held field, by kind, in the order first needed."""
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


def _types(cls_info):
    return cls_info.holds.types


def pxd_declarations(cls_info):
    """``cdef`` lines of the class's .pxd for its held fields."""
    lines = []
    for field, kind in _held(cls_info):
        lines += [f'cdef {_types(cls_info)[field][0]} {field}\n', f'cdef bint {field}_ready\n']
    return lines


def py_declarations(cls_info):
    """Annotated declarations of the class body of the .py for its held fields."""
    lines = []
    for field, kind in _held(cls_info):
        lines += [f'{field}: {_types(cls_info)[field][1]}\n', f'{field}_ready: _mx_cy.bint\n']
    return lines
