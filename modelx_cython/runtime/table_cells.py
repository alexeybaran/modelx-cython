"""Python access to the table cells and input References of a compiled model (api-tables, D130).

A compiled table cell keeps its internal table (a ``NativeTable``), which compiled code reads
natively and every thread shares without locks. Python callers see it through a per-name
property generated on the Space class (D130, superseding the read-only frames and the automatic
clears of D101, D109, D119 and D127):

- ``space.cell()`` and ``space.cell[()]`` return the plain pandas DataFrame the internal table was
  built from (the formula's own result, or the assigned frame), the same object on every call, as
  live modelx returns the cached value. It may be edited in place.
- ``space.cell[()] = df`` and ``space.cell = df`` convert ``df`` with the provider's
  ``table_from_value`` and store it as an input that later clears keep. Nothing is cleared.
- assigning an input Reference stores the value. Nothing is cleared.
- two clears, named as in live modelx: ``Model.clear_all()`` and ``Space.clear_all()`` (this Space and
  the Spaces below it: values, assigned inputs, ItemSpaces). Live's other clears (``Space.clear_cells``,
  ``clear_items``, ``clear_at``; ``cell.clear``, ``clear_all``, ``clear_at``) are not provided and raise
  NotImplementedError naming the two: without a dependency graph they would leave dependents stale.
  ``del space[key]`` deletes one ItemSpace, as before (D114). Every Space not cleared keeps its values; its
  table cells are rebuilt, on the next read, from the frame they hold, so an in-place edit made before
  a clear is what compiled code reads afterwards.

Assignment and clear raise RuntimeError while a ``collect`` of the model is running, at any number
of workers. An assignment made while another thread is calculating a single cell outside ``collect``
is not synchronised (as in live modelx, which is not thread-safe at all).
"""

import threading

_lock = threading.Lock()
_NO_FRAME = object()
_running = {}       # id(model) -> number of collect calls in progress
_clearing = {}      # id(model) -> number of clear_all() calls in progress
_inputs = {}        # id(model) -> {(id(space), name): (space, name, table)}


def collect_begin(model):
    with _lock:
        if _clearing.get(id(model)):
            raise RuntimeError('cannot run a collect while clear_all() or a table assignment of this model is '
                               'running; run it before or after')
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
    """Clear every cached value of ``model``, keep the assigned table inputs (a model-level clear).

    Every Space's caches are emptied and its ItemSpaces dropped and released (they are created again,
    with the current References, on the next access); an ItemSpace object a caller still holds is
    detached from the model.
    """
    for space in model._mx_walk(skip_self=True):
        space._mx_clear_values()
    _reapply_inputs(model)


def _reapply_inputs(model, spaces=None):
    with _lock:
        inputs = list(_inputs.get(id(model), {}).values())
    for space, name, table in inputs:
        if spaces is None or any(space is cleared for cleared in spaces):
            space._mx_set_table_cell(name, table)


def assign_table_cell(space, name, path, table):
    """Make ``table`` (already converted) the value of the table cell ``name`` of ``space``, as an
    input that clears keep. Values already computed from the old table stay until the user clears
    the Spaces that depend on it (D130)."""
    model = space._model
    _check_idle(model, f'the assignment of {path}')
    with _lock:
        _inputs.setdefault(id(model), {})[id(space), name] = (space, name, table)
    space._mx_set_table_cell(name, table)


def before_reference(space, path):
    """Raise if the Reference ``path`` cannot be assigned now (a collect of the model is running).

    The generated setter then stores the value. It does not clear: computed values stay, as in the
    exported model, until the user clears (D130).
    """
    _check_idle(space._model, f'the assignment of {path}')


def _drop_inputs(model, spaces):
    with _lock:
        inputs = _inputs.get(id(model), {})
        for key, (owner, cell, _) in list(inputs.items()):
            if any(owner is space for space in spaces):
                del inputs[key]


def clear_all(space):
    """``Space.clear_all()`` as in live modelx (D130): the values and assigned inputs of the cells of this
    Space and the Spaces below it are cleared and their ItemSpaces dropped (released through P2). Every
    other cell keeps its value: the compiled model holds no dependency graph, so clearing what depends on
    a change is the user's call, as on the exported model. The table cells kept are rebuilt from the
    frames they hold on their next read, so an in-place edit is what compiled code reads. Refused while a
    collect of the model is running."""
    model = space._model
    # the check and the mark are one step under the lock: a collect cannot start between them (collect_begin
    # refuses while the model is being cleared), so no worker reads a cache while it is cleared
    with _lock:
        if _running.get(id(model)):
            raise RuntimeError(f'cannot run {type(space).__name__[3:]}.clear_all() while a collect of this '
                               'model is running; do it before or after the run')
        _clearing[id(model)] = _clearing.get(id(model), 0) + 1
    try:
        cleared = list(space._mx_walk())
        _drop_inputs(model, cleared)
        for target in cleared:
            target._mx_clear_values()
        rebuild_tables(model, cleared)
        forget_prepared()
    finally:
        with _lock:
            count = _clearing.get(id(model), 0) - 1
            if count > 0:
                _clearing[id(model)] = count
            else:
                _clearing.pop(id(model), None)


# T-3: the PreparedTable caches of the loaded models (runtime _tables, one per compiled model); a clear drops
# them, so a frame edited in place is prepared again on its next read
_prepared_caches = []


_binders = {}       # top-level package of a compiled model -> its runtime _tables.bind


def register_prepared(forget, bind=None, package=None):
    with _lock:
        _prepared_caches.append(forget)
        if bind is not None:
            _binders[package] = bind


def assigned(space, name):
    """Whether the cell ``name`` of ``space`` holds a value the user assigned (an input), not its formula's."""
    with _lock:
        return (id(space), name) in _inputs.get(id(space._model), {})


def bind_frame(space, value):
    """T-4: prepare ``value`` (a frame cell's computed or assigned frame) in the model's PreparedTable cache, so
    the first fast read finds it prepared. A value that is not a preparable frame is left as it is."""
    bind = _binders.get(type(space).__module__.split('.')[0])
    if bind is not None:
        bind(value)


def forget_prepared():
    with _lock:
        caches = list(_prepared_caches)
    for forget in caches:
        forget()


CLEARS = ("Clear the Space that depends on the change (Space.clear_all()) or the whole model "
          "(Model.clear_all()); a compiled model has no dependency graph, so no other clear is provided.")
CELL_CLEAR = "{path}.{method}() is not provided by a compiled model. " + CLEARS


def not_provided(space, method):
    """The other clears of live modelx's Space (clear_cells, clear_items, clear_at) are not provided."""
    raise NotImplementedError(f'{type(space).__name__[3:]}.{method}() is not provided by a compiled model. '
                              + CLEARS)


def rebuild_tables(model, skip=()):
    """Mark every table cell of ``model`` outside ``skip`` for a rebuild from the frame it holds (D130):
    the next read converts that frame again instead of running the formula, so an in-place edit of
    the frame reaches compiled code. Values computed from the old table are not touched."""
    for target in model._mx_walk(skip_self=True):
        if not any(target is cleared for cleared in skip):
            stale = getattr(target, '_mx_stale_tables', None)
            if stale is not None:
                stale()


def rebind_source(table):
    """What a table cell marked by :func:`rebuild_tables` rebinds from: the frame it holds (the source
    frame, or the frame handed to Python), else the internal table itself (nothing to rebuild)."""
    frame = getattr(table, '_mx_python_frame', None)
    if frame is not None and type(frame).__module__.split('.')[0] == 'pandas':
        return frame
    from modelx_helpers.immutable_inputs import table_frame
    owner = getattr(table, '__modelx_table_owner__', None)
    frame = table_frame(owner()) if owner is not None else None
    return frame if frame is not None else table


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
    """What a Python caller gets for an internal table, the same object on every request, equal to what
    live modelx returns (D130): the plain DataFrame the table was admitted from (a formula's result or
    an assigned frame), which may be edited in place (compiled code reads the edit after a clear); for a
    table the model text builds with the helper (``pdn.table``), the helper Table itself, as live.

    Only the standard provider's ``NativeTable`` has a frame; another provider's object (bound by its own
    result binder) is returned as it is, as before D101.
    """
    frame = getattr(table, '_mx_python_frame', _NO_FRAME)
    if frame is _NO_FRAME:
        return table
    if frame is None:
        with _lock:
            frame = table._mx_python_frame
            if frame is None:
                from modelx_helpers.immutable_inputs import table_frame
                owner = table.__modelx_table_owner__()
                frame = table_frame(owner)          # a table admitted from a frame
                if frame is None:
                    frame = owner                   # a helper Table: what the formula returned, as live
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

    def _mx_no_cell_clear(self, method):
        raise NotImplementedError(CELL_CLEAR.format(path=self._path, method=method))

    def clear(self):
        """Not provided (D130): live modelx's cell-level clear; use the Space or model clear."""
        self._mx_no_cell_clear('clear')

    def clear_all(self):
        """Not provided (D130): live modelx's cell-level clear; use the Space or model clear."""
        self._mx_no_cell_clear('clear_all')

    def clear_at(self, *args, **kwargs):
        """Not provided (D130): live modelx's cell-level clear; use the Space or model clear."""
        self._mx_no_cell_clear('clear_at')

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'<compiled table cell {self._path}>'


def row_series(record, path):
    """What a Python caller gets for a cell whose value is a row of a table admitted from a frame: the
    plain pandas row (``frame.loc[key]``, as live gives it), the same object on every call of the
    cached cell (D130, superseding D119's read-only Series). Any other value is returned as it is.
    Compiled code keeps reading the Record."""
    from modelx_helpers.immutable_inputs import record_row
    found = record_row(record)
    if found is None:
        return record
    row, cached = found
    if cached is None:
        with _lock:
            cached = record_row(record)[1]
            if cached is None:
                cached = row()
                object.__setattr__(record, '_frame_row', (row, cached))
    return cached
