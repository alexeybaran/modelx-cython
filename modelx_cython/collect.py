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
from .consts import FORMULA_PREF, MODULE_PREF, MX_SPACE_MOD, SPACE_PREF
from .syntax import arguments, code, dotted_name, literal_value, same_expression, single_statement, walk

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


def _reads_parents(class_node, cells):
    """Whether a cached, uncached or Space formula reads its enclosing objects."""
    found = []

    class Visitor(cst.CSTVisitor):
        def visit_Attribute(self, node):
            if dotted_name(node) in ('self._model', 'self._parent', 'self._space'):
                found.append(node)

    for item in class_node.body.body:
        if not isinstance(item, cst.FunctionDef):
            continue
        name = item.name.value
        if name.startswith(FORMULA_PREF) or name == '_mx_space_formula' or (name in cells and name != '__call__'):
            item.visit(Visitor())
    return bool(found)


def member_specs(class_node, cls_info, modules):
    """Names of References and parent access in the collected Space and its named subtree."""
    members = []
    pending = [((), class_node, cls_info)]
    while pending:
        path, node, info = pending.pop()
        members.append((path, tuple(sorted(info.refs)), _reads_parents(node, info.cells),
                        _local_ref_copies(node, len(path))))
        if info.spaces:
            package = info.module.fqname.rsplit('.', 1)[0]
            child_module = modules.get(f'{package}.{MODULE_PREF}{info.name[len(SPACE_PREF):]}.{MX_SPACE_MOD}')
            if child_module is None:
                return tuple(members), False
            for child in child_module.visitor.wrapper.module.body:
                if isinstance(child, cst.ClassDef) and child.name.value in child_module.classes:
                    pending.append(((*path, child.name.value[len(SPACE_PREF):]),
                                    child, child_module.classes[child.name.value]))
    return tuple(members), True


def _local_ref_copies(class_node, depth):
    """Find relative References that the generated copy can bind inside the private tree."""
    method = next(item for item in class_node.body.body
                  if isinstance(item, cst.FunctionDef) and item.name.value == '_mx_copy_refs')
    candidates = []
    for line in method.body.body:
        assign = single_statement(line)
        if not (isinstance(assign, cst.Assign) and len(assign.targets) == 1
                and isinstance(assign.value, cst.IfExp)):
            continue
        target = dotted_name(assign.targets[0].target)
        if target is None or not target.startswith('self.'):
            continue
        ref = target[len('self.'):]
        value = assign.value
        base = f'base.{ref}'
        if dotted_name(value.orelse) != base:
            continue
        test = arguments(value.test, count=1)
        if test is None or dotted_name(test[0]) != 'base_root':
            continue
        guard = dotted_name(value.test.func)
        if guard not in (f'{base}._mx_is_in', f'{base}.__self__._mx_is_in'):
            continue
        endpoint = (dotted_name(value.body) or '').split('.')
        if endpoint[0] == 'self' and endpoint.count('_parent') <= depth:
            candidates.append((ref, '.__self__.' in guard))
    return tuple(candidates)


def _dynamic_refs(method, refs):
    """Find the named Reference slots accepted by the exported Space formula wrapper."""
    for node in walk(method):
        if (isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.Subtract)
                and arguments(node.left, 'refs.keys', 0) == () and isinstance(node.right, cst.Tuple)):
            try:
                names = literal_value(node.right)
            except ValueError:
                break
            if all(isinstance(name, str) for name in names):
                return names
    return refs


def _item_constructor(item):
    """Recognize the detached constructor, including the old export's leading Direct lookup."""
    if not isinstance(item.body, cst.IndentedBlock) or not item.params.params:
        return None
    params = item.params.params[1:]
    args = ', '.join(param.name.value for param in params)
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
    returned = cst.parse_statement('return _mx_root\n')
    if not same_expression(body[:2], constructor[:2]) or not same_expression(body[-1:], (returned,)):
        return None
    build = body[2:-1]
    if len(build) == 1 and isinstance(protected := build[0], cst.Try):
        cleanup = cst.parse_statement(
            'try:\n'
            '    pass\n'
            'except BaseException:\n'
            '    _mx_sys.release_collect_item(_mx_root)\n'
            '    raise\n')
        if not same_expression(protected.with_changes(body=cleanup.body), cleanup):
            return None
        build = protected.body.body
    if not same_expression(build[:1], constructor[2:]):
        return None
    initialize = build[1:]
    apply = cst.parse_statement(f'_mx_root._mx_apply_formula({args})\n')
    has_formula = same_expression(initialize[:1], (apply,))
    key = f'({args},)' if len(params) == 1 else f'({args})'
    apply_inputs = cst.parse_statement(f'self._mx_apply_item_inputs(_mx_root, {key})\n')
    remainder = initialize[has_formula:]
    has_inputs = same_expression(remainder, (apply_inputs,))
    if remainder and not has_inputs:
        return None
    return args, constructor, initialize, has_formula, has_inputs


def _private_constructor(item, constructor, initializers, has_formula, aliases):
    """Give the generic constructor a lock for its private tree before running its formula."""
    base, root, walk = constructor
    lock = cst.parse_module(
        'if _mx_s._mx_lock is not None:\n'
        '    if _mx_private_lock is None:\n'
        '        _mx_private_lock = _mx_collect_rlock()\n'
        '    _mx_s._mx_lock = _mx_private_lock\n').body
    walk = walk.with_changes(body=walk.body.with_changes(body=(*lock, *walk.body.body)))
    initialize = code(walk)
    if has_formula:
        initialize += 'with _mx_collect_factory_guard(self):\n' + textwrap.indent(code(initializers[0]), '    ')
        initializers = initializers[1:]
    initialize += ''.join(code(stmt) for stmt in initializers)
    body = cst.parse_module(
        f'{code(base)}{code(root)}'
        '_mx_private_lock = None\n'
        'try:\n'
        f'{textwrap.indent(initialize, "    ")}'
        'except BaseException:\n'
        f'    _mx_release_collect_tree(_mx_root, {aliases!r})\n'
        '    _mx_root = _mx_s = None\n'
        '    raise\n'
        'return _mx_root\n').body
    return item.body.with_changes(body=body)


def hook(class_node, cls_name, members, complete, dots='.', fast=True, reuse=False):
    """The compiled detached constructor and hook for a supported exported Space, or None.

    With ``fast``, a new item of a Space without child Spaces or roots is built on the typed
    class. Other items use the export's constructor."""
    methods = {item.name.value: item for item in class_node.body.body if isinstance(item, cst.FunctionDef)}
    item = methods.get('_mx_collect_item')
    calls = [call for call in map(_collect_call, methods.values()) if call is not None]
    if item is None or len(calls) != 1:
        return None
    constructor = _item_constructor(item)
    if constructor is None:
        raise ValueError(f'{cls_name[3:]}: unsupported exported Collect item constructor')
    params, build, initializers, apply_formula, apply_inputs = constructor
    if apply_inputs:
        fast = reuse = False
    aliases = tuple((path, copies) for path, _, _, copies in members if copies)
    runtime_members = tuple(member[:3] for member in members)
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
    apply = ''
    dynamic_refs = ()
    if apply_formula:
        apply = (
            f'                with _mx_collect_factory_guard(self):\n'
            f'                    _mx_fast._mx_apply_formula({params})\n')
        dynamic_refs = _dynamic_refs(methods['_mx_apply_formula'], members[0][1])
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
        clear_aliases = (f'                    _mx_clear_collect_refs(_mx_fast, {aliases!r})\n'
                         if aliases else '')
        built = (
            f'        if type(self) is {cls_name} and not self._mx_spaces and not self._mx_roots:\n'
            f'{initialize}'
            f'            try:\n'
            f'                _mx_fast._mx_lock = _mx_private_lock\n'
            f'                _mx_fast._mx_copy_refs(self, self)\n'
            f'                _mx_fast._mx_roots.append(_mx_fast)\n'
            f'                self._mx_assign_params(_mx_fast, {params})\n'
            f'{apply}'
            f'                if _mx_spare[2] is not None:\n'
            f'                    _mx_check_collect_bindings(_mx_fast, _mx_spare[2])\n'
            f'                _mx_out = _mx_prepare_output(_mx_fast.collect_outputs(), _mx_live_types)\n'
            f'{mark_prepared}'
            f'                return _mx_out\n'
            f'            finally:\n'
            f'                if _mx_fast._mx_itemspaces:\n'
            f'                    _mx_release_collect_tree(_mx_fast, {aliases!r})\n'
            f'                else:\n'
            f'{clear_aliases}'
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
        f'            if _mx_spare[2] is not None:\n'
        f'                _mx_check_collect_bindings(_mx_root, _mx_spare[2])\n'
        f'            return _mx_prepare_output(_mx_root.collect_outputs(), _mx_live_types)\n'
        f'        finally:\n'
        f'            _mx_release_collect_tree(_mx_root, {aliases!r})\n'
        f'            _mx_root = None\n'
        f'    return _mx_collect_evaluate(self, _mx_coords, {code(outputs_at)},\n'
        f'                                {runtime_members!r}, {complete!r}, {apply_formula!r}, {dynamic_refs!r})\n')
    return _private_constructor(item, build, initializers, apply_formula, aliases), cst.parse_module(source).body


def reuses_items(visitor, cls_name):
    """Whether collect reuses one item per worker: a collect Space without dynamic References (whose
    ``__dict__`` is no declared attribute)."""
    return (cls_name in visitor.collect_spaces and cls_name not in visitor.dynamic_ref_classes
            and '_mx_input_items' not in visitor.input_fields.get(cls_name, ()))


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
from contextlib import nullcontext as _mx_collect_nullcontext
from threading import RLock as _mx_collect_rlock
from types import BuiltinMethodType as _mx_collect_builtin_method, MethodType as _mx_collect_method


def _mx_collect_factory_guard(space):
    model_lock = space._model._mx_lock
    return model_lock if model_lock is not None else _mx_collect_nullcontext()


def _mx_collect_ref_owner(value):
    if isinstance(value, (_mx_collect_builtin_method, _mx_collect_method)):
        return value.__self__
    return value


def _mx_clear_collect_refs(root, aliases):
    by_path = dict(aliases)
    owned = set()
    candidates = []
    pending = [((), root)]
    member: _mx_sys.BaseSpace
    while pending:
        path, member = pending.pop()
        owned.add(id(member))
        names = by_path.get(path)
        if names:
            candidates.append((member, names))
        pending.extend(((*path, name), child) for name, child in member._mx_spaces.items())
        if member._mx_itemspaces:
            pending.extend((path, child) for child in member._mx_itemspaces.values())
    for member, names in candidates:
        for name, bound_method in names:
            value = getattr(member, name, None)
            if bound_method:
                value = _mx_collect_ref_owner(value)
            if id(value) in owned:
                setattr(member, name, None)


def _mx_release_collect_tree(root: _mx_sys.BaseSpace, aliases):
    if aliases:
        _mx_clear_collect_refs(root, aliases)
    # Walk only owned child Spaces and their locally retained items, never References.
    items = [root]
    member: _mx_sys.BaseSpace
    while items:
        member = items.pop()
        items.extend(member._mx_spaces.values())
        member._mx_spaces.clear()
        if member._mx_itemspaces:
            items.extend(member._mx_itemspaces.values())
            member._mx_itemspaces.clear()
        member._mx_cells.clear()
        member._mx_roots.clear()
        member._space = None
        member._parent = None


def _mx_collect_member_at(space, path):
    for part in path:
        space = space._mx_spaces[part]
    return space


def _mx_unlocked_collect_ref(member, ref):
    owner = _mx_collect_ref_owner(getattr(member, ref, None))
    if isinstance(owner, _mx_sys.BaseSpace) and owner._mx_lock is None:
        return owner


def _mx_collect_ref_error(name, workers, path, ref):
    reference = '.'.join((*path, ref))
    return ValueError(f'collect of {{name}} with {{workers}} workers: the Reference {{reference}} is a Space '
                      'without a lock; export it in locked_spaces or use one worker')


def _mx_collect_preflight(space, members, workers, dynamic_refs):
    # A Reference into the base tree can be rebound to the private tree, depending on its refmode.
    name = space._name
    declared = [(path, _mx_collect_member_at(space, path), refs, reads_parents)
                for path, refs, reads_parents in members]
    base_members = {{id(member) for _, member, _, _ in declared}}
    recheck = {{(): set(dynamic_refs)}} if dynamic_refs else {{}}
    for path, member, refs, reads_parents in declared:
        if reads_parents:
            source = '.'.join((name, *path))
            raise ValueError(f'collect of {{name}} with {{workers}} workers: a formula of {{source}} reads _model, '
                             '_parent or _space, which the workers could fill at once; use one worker')
        for ref in refs:
            value = _mx_unlocked_collect_ref(member, ref)
            if value is not None:
                if id(value) not in base_members:
                    raise _mx_collect_ref_error(name, workers, path, ref)
                recheck.setdefault(path, set()).add(ref)
    if recheck:
        return name, workers, tuple((path, tuple(sorted(refs))) for path, refs in recheck.items())


def _mx_check_collect_bindings(root, checks):
    name, workers, members = checks
    private_members = None
    for path, refs in members:
        member = _mx_collect_member_at(root, path)
        for ref in refs:
            value = _mx_unlocked_collect_ref(member, ref)
            if value is None:
                continue
            if private_members is None:
                private_members = {{id(item) for item in root._mx_walk()}}
            if id(value) not in private_members:
                raise _mx_collect_ref_error(name, workers, path, ref)


def _mx_collect_evaluate(space, coords, outputs_at, members, complete, applies_formula, dynamic_refs):
    from {dots}_mx_collect import Accumulator
    import sys
    workers = getattr(space._model, '_mx_collect_workers', {workers!r})
    if type(workers) is not int or workers < 1:
        raise ValueError('_mx_collect_workers must be a positive int, not %r' % (workers,))
    accumulator = Accumulator()
    checks = None
    if workers > 1 and len(coords) > 1:
        checks = _mx_collect_preflight(space, members, workers, dynamic_refs)
    model_lock = getattr(space._model, '_mx_lock', None)
    if (workers == 1 or len(coords) == 1 or getattr(sys, '_is_gil_enabled', lambda: True)()
            or not complete or applies_formula and model_lock is None
            or model_lock is not None and getattr(model_lock, '_is_owned', lambda: True)()):
        # A locked caller cannot wait for workers that may need its shared lock.
        spare = [None, None, checks]
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

    def work():
        spare = [None, None, checks]     # reusable item, its lock and optional checks
        try:
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
