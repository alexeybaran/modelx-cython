"""Python access to the table cells and input References of a compiled model (api-tables, D101).

A compiled table cell keeps its internal table (a ``NativeTable``), which compiled code reads
natively and every thread shares without locks. Python callers see it through a per-name
property generated on the Space class:

- ``space.cell()`` and ``space.cell[()]`` return a read-only pandas DataFrame
  (:class:`ReadOnlyFrame`), one object per internal table, so every call returns the same
  object. Any in-place write raises TypeError naming the cell and the fix.
- ``space.cell[()] = df`` and ``space.cell = df`` convert ``df`` with the provider's
  ``table_from_value`` (the same conversion the compiled cell uses for its formula result),
  check it against the build-time schema, then clear **every cached value of the model**
  (a superset of the dependants: the compiled model holds no dependency graph) and store the
  new table as an input that later clears keep.
- assigning an input Reference (``space.ref = value``) clears the model in the same way.

Both raise RuntimeError while a ``collect`` of the model is running, at any number of workers.
An assignment made while another thread is calculating a single cell outside ``collect`` is
not synchronised (as in live modelx, which is not thread-safe at all).
"""

import threading

_lock = threading.Lock()
_running = {}       # id(model) -> number of collect calls in progress
_inputs = {}        # id(model) -> {(id(space), name): (space, name, table)}

MESSAGE = ("{cell} is a table input of the compiled model and is read-only. Build a new table "
           "and assign it, e.g.: t = {cell}().copy(); edit t; {cell}[()] = t")


def collect_begin(model):
    with _lock:
        _running[id(model)] = _running.get(id(model), 0) + 1


def collect_end(model):
    with _lock:
        count = _running.get(id(model), 0) - 1
        if count > 0:
            _running[id(model)] = count
        else:
            _running.pop(id(model), None)


def _check_idle(model, what):
    with _lock:
        if _running.get(id(model)):
            raise RuntimeError(f'cannot run {what} while a collect of this model is running; '
                               'do it before or after the run')


def clear_model(model):
    """Clear every cached value of ``model``, keep the assigned table inputs.

    Every Space's caches are emptied and its ItemSpaces dropped (they are created again, with
    the current References, on the next access); an ItemSpace object a caller still holds is
    detached from the model. This clears more than the dependants of what changed, which the
    compiled model cannot list.
    """
    for space in model._mx_walk(skip_self=True):
        space._mx_clear_values()
    with _lock:
        inputs = list(_inputs.get(id(model), {}).values())
    for space, name, table in inputs:
        space._mx_set_table_cell(name, table)


def assign_table_cell(space, name, path, table):
    """Make ``table`` (already converted) the value of the table cell ``name`` of ``space``."""
    model = space._model
    if space._mx_roots:
        raise NotImplementedError(
            f'{path}: a table cell of an ItemSpace cannot be assigned in a compiled model '
            '(assigning clears the model, which drops its ItemSpaces); assign the input it is '
            'built from on the Space that holds it')
    _check_idle(model, f'the assignment of {path}')
    with _lock:
        _inputs.setdefault(id(model), {})[id(space), name] = (space, name, table)
    clear_model(model)


def before_reference(space, path):
    """Raise if the Reference ``path`` cannot be assigned now (a collect of the model is running).

    The generated setter then stores the value and calls :func:`clear_model`.
    """
    _check_idle(space._model, f'the assignment of {path}')


def clear_all(space, name=None):
    """``clear_all()`` of a compiled Space (``name`` None) or of one of its table cells.

    As in live modelx, the assigned table inputs of that Space (and the Spaces below it) or of
    that cell are dropped, so their formulas run again; then every cached value of the model is
    cleared: the compiled model holds no dependency graph, so the clear is never limited to
    the Space and its dependants (a superset of live modelx's clear). Refused while a collect of
    the model is running.
    """
    model = space._model
    _check_idle(model, f'{type(space).__name__[3:]}.clear_all()' if name is None else f'{name}.clear_all()')
    with _lock:
        inputs = _inputs.get(id(model), {})
        for key, (owner, cell, _) in list(inputs.items()):
            if (owner is space or owner._mx_is_in(space)) and (name is None or cell == name):
                del inputs[key]
    clear_model(model)


_enums = {}


def enum_member(module, qualname, value):
    """The member of the IntEnum ``module.qualname`` whose value is ``value`` (D103).

    A compiled cell holds an admitted enum as its C integer; this is what a Python caller,
    an object-valued use or a collected output receives, as in live modelx.
    """
    cls = _enums.get((module, qualname))
    if cls is None:
        import importlib
        cls = importlib.import_module(module)
        for part in qualname.split('.'):
            cls = getattr(cls, part)
        _enums[module, qualname] = cls
    member = cls._value2member_map_.get(value)
    return member if member is not None else cls(value)


def frame_of(table, path):
    """The one read-only DataFrame of an internal table, created on first request."""
    frame = table._mx_python_frame
    if frame is None:
        with _lock:
            frame = table._mx_python_frame
            if frame is None:
                from modelx_helpers import pdn
                frame = ReadOnlyFrame._wrap(pdn.to_pandas(table), table, path)
                table._mx_python_frame = frame
    return frame


class TableCell:
    """What ``space.cell`` gives a Python caller for a compiled table cell."""
    __slots__ = ('_space', '_name', '_path')

    def __init__(self, space, name, path):
        self._space, self._name, self._path = space, name, path

    def __call__(self):
        return frame_of(self._space._mx_table_cell_value(self._name), self._path)

    def __getitem__(self, key):
        if key != ():
            raise KeyError(f'{self._path} has no parameters: use {self._path}[()]')
        return self()

    def __setitem__(self, key, value):
        if key != ():
            raise KeyError(f'{self._path} has no parameters: use {self._path}[()] = table')
        setattr(self._space, self._name, value)

    def clear_all(self):
        """Drop an assigned table (the formula runs again) and clear the model's values."""
        clear_all(self._space, self._name)

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'<compiled table cell {self._path}>'


def _ro_indexer():
    import pandas as pd

    class _Indexer:
        __slots__ = ('_inner', '_frame')

        def __init__(self, inner, frame):
            self._inner, self._frame = inner, frame

        def __getitem__(self, key):
            return self._inner[key]

        def __setitem__(self, key, value):
            self._frame._mx_refuse()

    return pd, _Indexer


_FRAME_CLASS = None


def _frame_class():
    global _FRAME_CLASS
    if _FRAME_CLASS is not None:
        return _FRAME_CLASS
    pd, _Indexer = _ro_indexer()
    inplace = set()
    for attr in dir(pd.DataFrame):
        code = getattr(getattr(pd.DataFrame, attr, None), '__code__', None)
        if not attr.startswith('_') and code is not None and 'inplace' in code.co_varnames:
            inplace.add(attr)

    class ReadOnlyFrame(pd.DataFrame):
        """A read-only DataFrame of a compiled table cell. Anything derived is plain pandas."""
        _metadata = ['_mx_table', '_mx_path']

        @property
        def _constructor(self):
            return pd.DataFrame

        @classmethod
        def _wrap(cls, frame, table, path):
            obj = cls(frame)
            for block in obj._mgr.blocks:   # .values / .to_numpy() views cannot write either
                values = block.values
                array = values if hasattr(values, 'flags') else getattr(values, '_ndarray', None)
                if array is not None and hasattr(array, 'flags'):
                    array.flags.writeable = False   # extension arrays without one stay guarded by the indexers
            object.__setattr__(obj, '_mx_table', table)
            object.__setattr__(obj, '_mx_path', path)
            return obj

        def __modelx_table_cell_source__(self):
            return self._mx_table

        def __modelx_table_owner__(self):
            """The helper Table behind the frame, so helper functions (``pdn.to_pandas``, ...) accept it."""
            return self._mx_table.__modelx_table_owner__()

        def _mx_refuse(self, *args, **kwargs):
            raise TypeError(MESSAGE.format(cell=self._mx_path))

        loc = property(lambda self: _Indexer(pd.DataFrame.loc.__get__(self), self))
        iloc = property(lambda self: _Indexer(pd.DataFrame.iloc.__get__(self), self))
        at = property(lambda self: _Indexer(pd.DataFrame.at.__get__(self), self))
        iat = property(lambda self: _Indexer(pd.DataFrame.iat.__get__(self), self))
        __setitem__ = __delitem__ = insert = pop = _mx_refuse

        def __setattr__(self, name, value):
            if name in ('index', 'columns') or (name in getattr(self, 'columns', ())):
                self._mx_refuse()
            super().__setattr__(name, value)

        def __getattribute__(self, name):
            attr = super().__getattribute__(name)
            if name in inplace:
                def guarded(*args, **kwargs):
                    if kwargs.get('inplace'):
                        self._mx_refuse()
                    return attr(*args, **kwargs)
                return guarded
            return attr

    _FRAME_CLASS = ReadOnlyFrame
    return ReadOnlyFrame


class _FrameFactory:
    """``ReadOnlyFrame`` resolved on first use, so importing this module does not import pandas."""

    def __getattr__(self, name):
        return getattr(_frame_class(), name)

    def __instancecheck__(self, obj):
        return isinstance(obj, _frame_class())


ReadOnlyFrame = _FrameFactory()
