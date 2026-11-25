"""Python access to the table cells of a compiled model (api-tables).

A compiled table cell keeps the value its formula returned: a plain pandas DataFrame, which compiled code reads
through the table prepared from it (``runtime._tables``), or a native table. Python callers see it through a
per-name property generated on the Space class: ``space.cell()`` and ``space.cell[()]`` return the value, the
same object on every call, as live modelx returns the cached value.

As upstream modelx-cython, a compiled model offers no clear and no assignment of a table cell; its References
are plain attributes.
"""

import threading

_lock = threading.Lock()
_binders = {}       # top-level package of a compiled model -> its runtime _tables.bind


def register_prepared(bind=None, package=None):
    with _lock:
        if bind is not None:
            _binders[package] = bind


def bind_frame(space, value):
    """T-4: prepare ``value`` (a frame cell's computed or assigned frame) in the model's PreparedTable cache and
    return what an item reading it holds (T-4.2): its PreparedTable, or a table whose every slot is -1 (not a
    preparable frame); None when the model has no fast reads."""
    bind = _binders.get(type(space).__module__.split('.')[0])
    return bind(value) if bind is not None else None


class TableCell:
    """What ``space.cell`` gives a Python caller for a compiled table cell."""
    __slots__ = ('_space', '_name', '_path')

    def __init__(self, space, name, path):
        self._space, self._name, self._path = space, name, path

    def __call__(self):
        return self._space._mx_table_cell_value(self._name)

    def __getitem__(self, key):
        if key != ():
            raise KeyError(f'{self._path} has no parameters: use {self._path}[()]')
        return self()

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'<compiled table cell {self._path}>'
