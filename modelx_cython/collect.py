"""Compiled ``collect()``: the hook of modelx's exported collect, version 1.

A Space that declares collect exports a constructor for one coordinate and a method that calls
``_mx_sys.collect_from_space``. The compiled hook builds a private ItemSpace for each coordinate.
Locked Spaces in that tree get their own reentrant lock. The export's ``prepare_output`` detaches
and validates the outputs before the private item is released. A worker can reset and reuse an
unescaped item of a Space without child Spaces or dynamic References. The export's ``Accumulator``
adds the outputs in domain order at any number of workers.

Workers: the spec key ``collect_workers`` (default 1), or ``_mx_collect_workers`` set on the compiled
model object. With more than one, items run on that many threads; their outputs are added in domain
order as they complete, and an error raises the exception of the first failing coordinate, as one
worker does. With more than one worker, on every build, a Reference of the Space that is a Space without
a lock (not exported in ``locked_spaces``) raises ValueError naming it, as does a formula of the Space that
reads ``_model``, ``_parent`` or ``_space``: what those reach cannot be filled by two threads at once.
Threads run model code at the same time only on free-threaded Python; with the GIL, one worker runs.
"""
import pathlib
import re
import textwrap

import libcst as cst

from . import item_roots
from .syntax import arguments, code, dotted_name, same_expression, single_statement

HOOK_VERSION = 1


def _collect_call(method):
    """The ``collect_from_space`` call that an exported collect method returns, or None."""
    body = method.body.body if isinstance(method.body, cst.IndentedBlock) else ()
    stmt = body[-1] if body else None
    if not (isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1
            and isinstance(stmt.body[0], cst.Return)):
        return None
    call = stmt.body[0].value
    if (isinstance(call, cst.Call) and dotted_name(call.func) == '_mx_sys.collect_from_space'
            and len(call.args) == 5 and all(a.keyword is None and not a.star for a in call.args)
            and dotted_name(call.args[4].value) == 'self._mx_collect_cache'
            and isinstance(call.args[2].value, cst.Lambda)
            and isinstance(call.args[3].value, cst.Tuple)):
        return call
    return None


def _reads_parents(class_node):
    """Whether a formula of the class reads ``self._model``, ``self._parent`` or ``self._space``."""
    found = []

    class Visitor(cst.CSTVisitor):
        def visit_Attribute(self, node):
            if dotted_name(node) in ('self._model', 'self._parent', 'self._space'):
                found.append(node)

    for item in class_node.body.body:
        if isinstance(item, cst.FunctionDef) and item.name.value.startswith('_f_'):
            item.visit(Visitor())
    return bool(found)


def _item_constructor(item):
    """Recognize the detached constructor, including the old export's leading Direct lookup."""
    if not isinstance(item.body, cst.IndentedBlock) or not item.params.params:
        return None
    args = ', '.join(param.name.value for param in item.params.params[1:])
    body = item.body.body
    lookup = single_statement(body[0]) if body else None
    if isinstance(lookup, cst.Assign) and arguments(lookup.value, 'self._mx_itemspaces.get', 1):
        legacy = cst.parse_module(
            f'_mx_root = {code(lookup.value)}\n'
            'if _mx_root is not None:\n'
            '    return _mx_root\n').body
        if same_expression(body[:2], legacy):
            body = body[2:]
    constructor = cst.parse_module(
        '_mx_base = self\n'
        '_mx_root = _mx_base.__class__(self)\n'
        'for _mx_s, _mx_b in zip(_mx_root._mx_walk(), _mx_base._mx_walk()):\n'
        '    _mx_s._mx_copy_refs(_mx_b, _mx_base)\n'
        '    for _mx_r in self._mx_roots:\n'
        '        _mx_r._mx_copy_params(_mx_s)\n'
        f'    self._mx_assign_params(_mx_s, {args})\n'
        '    _mx_s._mx_roots.extend(self._mx_roots)\n'
        '    _mx_s._mx_roots.append(_mx_root)\n').body
    if not (same_expression(body[:3], constructor) and body[3:]):
        return None
    suffix = body[3:]
    returned = cst.parse_statement('return _mx_root\n')
    if same_expression(suffix, (returned,)):
        return args, constructor, False
    apply = cst.parse_statement(f'_mx_root._mx_apply_formula({args})\n')
    if same_expression(suffix, (apply, returned)):
        return args, constructor, True
    return None


def _private_constructor(item, constructor, params, apply_formula):
    """Give the generic constructor a lock for its private tree before running its formula."""
    base, root, walk = constructor
    lock = cst.parse_module(
        'if _mx_s._mx_lock is not None:\n'
        '    if _mx_private_lock is None:\n'
        '        _mx_private_lock = _mx_collect_rlock()\n'
        '    _mx_s._mx_lock = _mx_private_lock\n').body
    walk = walk.with_changes(body=walk.body.with_changes(body=(*lock, *walk.body.body)))
    initialize = code(walk)
    if apply_formula:
        initialize += f'_mx_root._mx_apply_formula({params})\n'
    body = cst.parse_module(
        f'{code(base)}{code(root)}'
        '_mx_private_lock = None\n'
        'try:\n'
        f'{textwrap.indent(initialize, "    ")}'
        'except BaseException:\n'
        '    _mx_release_collect_tree(_mx_root)\n'
        '    _mx_root = _mx_s = None\n'
        '    raise\n'
        'return _mx_root\n').body
    return item.body.with_changes(body=body)


def hook(class_node, cls_name, refs, dots='.', fast=True, reuse=False):
    """The compiled detached constructor and hook for a supported exported Space, or None.

    ``refs`` are the names of its References. With ``fast``, a new item of a Space without child
    Spaces or roots is built on the typed class. Other items use the export's constructor."""
    methods = {item.name.value: item for item in class_node.body.body if isinstance(item, cst.FunctionDef)}
    item = methods.get('_mx_collect_item')
    calls = [call for call in map(_collect_call, methods.values()) if call is not None]
    if item is None or len(calls) != 1:
        return None
    constructor = _item_constructor(item)
    if constructor is None:
        raise ValueError(f'{cls_name[3:]}: unsupported exported Collect item constructor')
    params, build, apply_formula = constructor
    call = calls[0]
    others = call.args[3].value
    lam = call.args[2].value
    if arguments(lam.body, 'self._mx_collect_item', len(item.params.params) - 1) is None:
        raise ValueError(f'{cls_name[3:]}: unsupported exported Collect item call')
    # the worker's slot first: the item it keeps between coordinates
    outputs_at = lam.with_changes(
        params=lam.params.with_changes(params=(cst.Param(cst.Name('_mx_spare')), *lam.params.params)),
        body=lam.body.with_changes(func=cst.Name('_mx_outputs'),
                                   args=(cst.Arg(cst.Name('_mx_spare')), *lam.body.args)))
    unpack = f'    {code(others)} = _mx_args\n' if others.elements else ''
    apply = f'                _mx_fast._mx_apply_formula({params})\n' if apply_formula else ''
    initialize = ''
    mark_prepared = ''
    retain = ''
    if reuse:
        initialize = (
            f'            _mx_fast: {cls_name} = _mx_spare[0]\n'
            f'            if _mx_fast is None:\n'
            f'                _mx_fast = {cls_name}(self)\n'
            f'                _mx_private_lock = _mx_collect_rlock() if _mx_fast._mx_lock is not None else None\n'
            f'            else:\n'
            f'                _mx_spare[0] = None\n'
            f'                _mx_private_lock = _mx_spare[1]\n'
            f'                _mx_spare[1] = None\n'
            f'                _mx_fast.__init__(self)\n'
            f'            _mx_prepared = False\n')
        mark_prepared = '                _mx_prepared = True\n'
        retain = (
            f'                    if _mx_prepared and _mx_refcount(_mx_fast) == 1:\n'
            f'                        {reset_root(cls_name)}(_mx_fast)\n'
            f'                        _mx_spare[0] = _mx_fast\n'
            f'                        _mx_spare[1] = _mx_private_lock\n')
    elif fast:
        initialize = (
            f'            _mx_fast: {cls_name} = {cls_name}(self)\n'
            f'            _mx_private_lock = _mx_collect_rlock() if _mx_fast._mx_lock is not None else None\n')
    if initialize:
        built = (
            f'        if type(self) is {cls_name} and not self._mx_spaces and not self._mx_roots:\n'
            f'{initialize}'
            f'            try:\n'
            f'                _mx_fast._mx_lock = _mx_private_lock\n'
            f'                _mx_fast._mx_copy_refs(self, self)\n'
            f'                _mx_fast._mx_roots.append(_mx_fast)\n'
            f'                self._mx_assign_params(_mx_fast, {params})\n'
            f'{apply}'
            f'                _mx_out = _mx_prepare_output(_mx_fast.collect_outputs(), _mx_live_types)\n'
            f'{mark_prepared}'
            f'                return _mx_out\n'
            f'            finally:\n'
            f'                if _mx_fast._mx_itemspaces:\n'
            f'                    _mx_release_collect_tree(_mx_fast)\n'
            f'                else:\n'
            f'                    {item_roots.release_root(cls_name)}(_mx_fast)\n'
            f'{retain}'
            f'                _mx_fast = None\n')
    else:
        built = ''
    source = (
        f'_mx_collect_hook_version = {HOOK_VERSION}\n'
        f'\n'
        f'def _mx_collect_hook(self, _mx_coords, _mx_args):\n'
        f'    from {dots}_mx_collect import prepare_output as _mx_prepare_output\n'
        f'    _mx_live_types = (_mx_sys.BaseMxObject,)\n'
        f'{unpack}'
        f'    def _mx_outputs(_mx_spare, {params}):\n'
        f'{built}'
        f'        _mx_root = self._mx_collect_item({params})\n'
        f'        try:\n'
        f'            return _mx_prepare_output(_mx_root.collect_outputs(), _mx_live_types)\n'
        f'        finally:\n'
        f'            _mx_release_collect_tree(_mx_root)\n'
        f'            _mx_root = None\n'
        f'    return _mx_collect_evaluate(self, _mx_coords, {code(outputs_at)},\n'
        f'                                {tuple(refs)!r}, {_reads_parents(class_node)!r})\n')
    return _private_constructor(item, build, params, apply_formula), cst.parse_module(source).body


def reuses_items(visitor, cls_name):
    """Whether collect reuses one item per worker: a collect Space without dynamic References (whose
    ``__dict__`` is no declared attribute)."""
    return cls_name in visitor.collect_spaces and cls_name not in visitor.dynamic_ref_classes


def reset_root(cls_name):
    return f'_mx_reset_root_{cls_name}'


_SCALARS = {'bint', 'char', 'signed char', 'short', 'int', 'long', 'long long', 'unsigned char',
            'unsigned short', 'unsigned int', 'unsigned long', 'unsigned long long', 'float', 'double',
            'Py_ssize_t', 'size_t'}
_OBJECTS = {'object', 'dict', 'list', 'str', 'tuple', 'set', 'frozenset', 'bytes'}


def _declared(pattern):
    """Names of the C structs or extension types (``pattern``) declared in the compiler's .pxd files."""
    names = set()
    for path in pathlib.Path(__file__).parent.rglob('*.pxd'):
        names.update(re.findall(pattern, path.read_text(encoding='utf-8'), re.M))
    return names


def _fields(block):
    """(declaration, type, name) of the attribute lines of a ``cdef class`` block."""
    for line in block.splitlines():
        words = line.split()
        if words[:1] == ['cdef'] and '(' not in line and words[1:2] != ['class'] and len(words) > 2:
            words = [w for w in words[1:] if w not in ('public', 'readonly')]
            yield line.strip(), ' '.join(words[:-1]), words[-1]


def _base_fields():
    source = (pathlib.Path(__file__).parent / '_mx_sys.pxd').read_text(encoding='utf-8')
    blocks = {}
    for part in source.split('\ncdef class ')[1:]:
        name = part.split('(')[0].split(':')[0].strip()
        blocks[name] = part.split('\n', 1)[1].split('\ncdef ')[0]
    return [field for base in ('BaseMxObject', 'BaseParent', 'BaseSpace') for field in _fields(blocks.get(base, ''))]


def reset_function(cls_name, block, classes, values):
    """The ``cdef inline`` function, for the class's .pxd, that returns an item to the state of a new object.

    Every attribute that the class block and its bases declare (the object's layout) is listed: an object
    or a memoryview to None, as ``tp_new`` leaves it; any C value, array or struct to zero bytes, as the allocator leaves
    it. The dense value arrays (``values``) are left unwritten behind their flags, as the class's allocator
    leaves them in a new object (``value_storage``: it zeroes every byte but those arrays). A declaration of another kind stops the build naming the attribute and its Cell."""
    structs = _declared(r'^cdef struct (\w+)')
    objects = _OBJECTS | set(classes) | _declared(r'^cdef class (\w+)')
    lines = []
    for line, ctype, name in [*_base_fields(), *_fields(block)]:
        base = ctype.split('[')[0].split('.')[-1].strip()
        if line in values:
            continue
        if ('[' not in ctype and (base in objects or base.startswith('_c_'))
                or re.fullmatch(r'(const )?[\w ]+\[:(, *:)*\]', ctype)):     # an object or a memoryview
            lines.append(f'    root.{name} = None\n')
        elif (base in _SCALARS or base in structs) and re.fullmatch(r'[\w. ]+(\[\d+\])*', ctype):
            lines.append(f'    memset(&root.{name}, 0, sizeof(root.{name}))\n')
        else:
            cell = name.removeprefix('_v_').removeprefix('_has_').removeprefix('_spill_')
            raise ValueError(f'{cls_name[3:]}: collect cannot reset {line!r} (Cell {cell}) between items')
    return ('from libc.string cimport memset\nfrom cpython.ref cimport Py_REFCNT as _mx_refcount\n\n'
            f'cdef inline void {reset_root(cls_name)}({cls_name} root) noexcept:\n' + ''.join(lines))


def evaluator(workers, dots='.'):
    """The module-level functions shared by the collect hooks of a module."""
    return cst.parse_module(f'''
from threading import RLock as _mx_collect_rlock


def _mx_release_collect_tree(root: _mx_sys.BaseSpace):
    # Walk only owned child Spaces and their locally retained items, never References.
    items = [root]
    member: _mx_sys.BaseSpace
    while items:
        member = items.pop()
        items.extend(member._mx_spaces.values())
        if member._mx_itemspaces:
            items.extend(member._mx_itemspaces.values())
            member._mx_itemspaces.clear()
        member._mx_cells.clear()
        member._mx_roots.clear()
        member._space = None
        member._parent = None


def _mx_collect_evaluate(space, coords, outputs_at, refs, reads_parents):
    from {dots}_mx_collect import Accumulator
    import sys
    workers = getattr(space._model, '_mx_collect_workers', {workers!r})
    if type(workers) is not int or workers < 1:
        raise ValueError('_mx_collect_workers must be a positive int, not %r' % (workers,))
    accumulator = Accumulator()
    if workers > 1 and len(coords) > 1:
        # refused on every build, whether or not the threads would run model code at once
        name = type(space).__name__[len('_c_'):]
        if reads_parents:
            raise ValueError(f'collect of {{name}} with {{workers}} workers: a formula of {{name}} reads _model, '
                             '_parent or _space, which the workers could fill at once; use one worker')
        for ref in refs:
            value = getattr(space, ref, None)
            if isinstance(value, _mx_sys.BaseSpace) and value is not space and value._mx_lock is None:
                raise ValueError(f'collect of {{name}} with {{workers}} workers: the Reference {{ref}} is a Space '
                                 'without a lock; export it in locked_spaces or use one worker')
    model_lock = getattr(space._model, '_mx_lock', None)
    if (workers == 1 or len(coords) == 1 or getattr(sys, '_is_gil_enabled', lambda: True)()
            or model_lock is not None and getattr(model_lock, '_is_owned', lambda: True)()):
        # A locked caller cannot wait for workers that may need its shared lock.
        spare = [None, None]       # reusable item and its private lock
        try:
            for c in coords:
                accumulator.add(c, outputs_at(spare, c))
        finally:
            spare.clear()
        return accumulator.result()
    import threading
    lock = threading.Lock()
    pending = {{}}
    state = [0, 0, len(coords), None]       # next to take, next to add, stop, the first error

    def work_items(spare):
        while True:
            with lock:
                i = state[0]
                if i >= state[2]:
                    return
                state[0] = i + 1
            try:
                out = outputs_at(spare, coords[i])
            except BaseException as error:
                with lock:
                    if i < state[2]:
                        state[2], state[3] = i, error
                continue
            with lock:
                pending[i] = out
                while state[1] < state[2] and state[1] in pending:
                    k = state[1]
                    try:
                        accumulator.add(coords[k], pending.pop(k))
                    except BaseException as error:
                        state[2], state[3] = k, error
                        break
                    state[1] = k + 1

    def work():
        spare = [None, None]
        try:
            work_items(spare)
        finally:
            spare.clear()

    threads = [threading.Thread(target=work) for _ in range(min(workers, len(coords)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if state[3] is not None:
        raise state.pop()
    return accumulator.result()
''').body
