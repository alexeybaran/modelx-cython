"""Compiled ``collect()``: the hook of modelx's exported collect, version 1.

A Space that declares collect is exported with ``_mx_collect_item`` (the ItemSpace at one coordinate:
an existing one, or a new root that is not stored) and a method that calls
``_mx_sys.collect_from_space`` (``_mx_collect_run``, which the Space's ``collect`` property returns;
an ItemSpace has no ``collect``, as in live modelx). The compiled class adds ``_mx_collect_hook``, which modelx calls
with the checked domain: each item is built (as ``__call__`` builds a new root, by ``item_roots``, for
a Space without child Spaces; else by ``_mx_collect_item``), its ``collect_outputs()`` read and checked
by the export's ``prepare_output``, and the item released when collect created it. A new item of a
Space without child Spaces or dynamic References is kept in its worker's slot when nothing else holds
it, returned to the state of a new object (``reset_function``: every declared attribute, from the
class's own .pxd block) and built again on the same memory for the next coordinate. Its References stay as copied, and are copied
again before a coordinate when one is no longer the one the Space holds (rebound during collect), so every
item reads the Space's References at its coordinate, as live modelx gives. The outputs are added
by the export's ``Accumulator`` in domain order, so the result is the one live modelx computes, bit
for bit, at any number of workers. An output that is only a ``sum`` of exact floats, with the keys the
Accumulator's total holds and floats there, is added into that total directly (``total + value``, as the
Accumulator adds, in domain order) without ``prepare_output``; any other output goes through both.

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

import libcst as cst

from . import item_roots
from .syntax import dotted_name, walk

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


def _code(node):
    return cst.Module(body=()).code_for_node(node)


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


def hook(class_node, cls_name, refs, fast=True, reuse=False):
    """Statements to add to the body of a compiled Space class that declares collect, or ().

    ``refs`` are the names of its References. With ``fast``, a new item of a Space without child
    Spaces or roots is built and released by the functions of ``item_roots``, as ``__call__`` builds
    it (``transformer._fast_item_call``); any other item through the exported ``_mx_collect_item``."""
    methods = {item.name.value: item for item in class_node.body.body if isinstance(item, cst.FunctionDef)}
    item = methods.get('_mx_collect_item')
    calls = [call for call in map(_collect_call, methods.values()) if call is not None]
    if item is None or len(calls) != 1:
        return ()
    first = item.body.body[0]
    lookup = first.body[0].value if isinstance(first, cst.SimpleStatementLine) else None
    if (not isinstance(lookup, cst.Call) or dotted_name(lookup.func) != 'self._mx_itemspaces.get'
            or len(lookup.args) != 1):
        return ()
    call = calls[0]
    params = ', '.join(p.name.value for p in item.params.params[1:])
    others = call.args[3].value
    lam = call.args[2].value
    # the worker's slot first: the item it keeps between coordinates
    outputs_at = lam.with_changes(
        params=lam.params.with_changes(params=(cst.Param(cst.Name('_mx_spare')), *lam.params.params)),
        body=lam.body.with_changes(func=cst.Name('_mx_outputs'),
                                   args=(cst.Arg(cst.Name('_mx_spare')), *lam.body.args)))
    unpack = f'    {_code(others)} = args\n' if others.elements else ''
    apply = ('            _mx_fast._mx_apply_formula(%s)\n' % params
             if '_mx_apply_formula' in _code(item) else '')
    if reuse:
        # the References the reset keeps are copied again only when one is no longer the Space's
        copy_refs = ('                _mx_fast._mx_copy_refs(self, self)\n' if kept_refs(class_node) is None else
                     f'                if not {refs_held(cls_name)}(_mx_fast, self):\n'
                     f'                    _mx_fast._mx_copy_refs(self, self)\n')
        # one cleared item per worker, in the worker's slot: built again as __call__ builds a root, on the same object
        built = (
            f'        if type(self) is {cls_name} and not self._mx_spaces and not self._mx_roots:\n'
            f'            _mx_fast: {cls_name} = _mx_spare[0]\n'
            f'            if _mx_fast is None:\n'
            f'                _mx_fast = {item_roots.new_root(cls_name)}(self)\n'
            f'            else:\n'
            f'                _mx_spare[0] = None\n'
            f'                _mx_fast.__init__(self)\n'
            f'{copy_refs}'
            f'                _mx_fast._mx_roots.append(_mx_fast)\n'
            f'            self._mx_assign_params(_mx_fast, {params})\n'
            f'{apply}'
            f'            _mx_out = _mx_fast.collect_outputs()\n'
            f'            _mx_owned = not _mx_fast._mx_itemspaces\n'
            f'            {item_roots.release_root(cls_name)}(_mx_fast)\n'
            f'            if _mx_owned and _mx_refcount(_mx_fast) == 1:     # held by nothing else\n'
            f'                {reset_root(cls_name)}(_mx_fast)\n'
            f'                _mx_spare[0] = _mx_fast\n'
            f'            return _mx_out\n')
    elif fast:
        built = (
            f'        if type(self) is {cls_name} and not self._mx_spaces and not self._mx_roots:\n'
            f'            _mx_fast: {cls_name} = {item_roots.new_root(cls_name)}(self)\n'
            f'            self._mx_assign_params(_mx_fast, {params})\n'
            f'{apply}'
            f'            _mx_out = _mx_fast.collect_outputs()\n'
            f'            {item_roots.release_root(cls_name)}(_mx_fast)\n'
            f'            return _mx_out\n')
    else:
        built = ''
    source = (
        f'_mx_collect_hook_version = {HOOK_VERSION}\n'
        f'\n'
        f'def _mx_collect_hook(self, coords, args):\n'
        f'{unpack}'
        f'    def _mx_outputs(_mx_spare, {params}):\n'
        f'        _mx_root = self._mx_itemspaces.get({_code(lookup.args[0].value)})\n'
        f'        if _mx_root is not None:\n'
        f'            return _mx_root.collect_outputs()\n'
        f'{built}'
        f'        _mx_root = self._mx_collect_item({params})\n'
        f'        _mx_out = _mx_root.collect_outputs()\n'
        f'        {item_roots.release_item(cls_name)}(_mx_root)\n'
        f'        return _mx_out\n'
        f'    return _mx_collect_evaluate(self, coords, {_code(outputs_at)},\n'
        f'                                {tuple(refs)!r}, {_reads_parents(class_node)!r})\n')
    return cst.parse_module(source).body


def reuses_items(visitor, cls_name):
    """Whether collect reuses one item per worker: a collect Space without dynamic References (whose
    ``__dict__`` is no declared attribute)."""
    return cls_name in visitor.collect_spaces and cls_name not in visitor.dynamic_ref_classes


def reset_root(cls_name):
    return f'_mx_reset_root_{cls_name}'


def refs_held(cls_name):
    return f'_mx_refs_held_{cls_name}'


def kept_refs(class_node):
    """(attribute, base attribute) of each Reference the exported ``_mx_copy_refs`` copies from the base, but the
    Space's parameters, which every coordinate assigns; None when a line has another form. A line that rebases
    a Reference (``self._parent.X if base.A._mx_is_in(base_root) else base.A``) is kept while it holds
    ``base.A``."""
    methods = {item.name.value: item for item in class_node.body.body if isinstance(item, cst.FunctionDef)}
    copy, item = methods.get('_mx_copy_refs'), methods.get('_mx_collect_item')
    if copy is None or item is None:
        return None
    params = {p.name.value for p in item.params.params[1:]}
    pairs = []
    for stmt in copy.body.body:
        assign = stmt.body[0] if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1 else None
        if isinstance(assign, cst.Pass):
            continue
        if not isinstance(assign, cst.Assign) or len(assign.targets) != 1:
            return None
        target, value = dotted_name(assign.targets[0].target) or '', assign.value
        if isinstance(value, cst.IfExp):
            test = value.test
            if not (isinstance(test, cst.Call) and dotted_name(test.func) == f'{dotted_name(value.orelse)}._mx_is_in'):
                return None
            value = value.orelse
        source = dotted_name(value) or ''
        if not (target.count('.') == 1 and target.startswith('self.') and source.count('.') == 1
                and source.startswith('base.')):
            return None
        if target[5:] not in params:
            pairs.append((target[5:], source[5:]))
    return tuple(pairs)


def reached_cells(class_node, cells):
    """The Cells (of ``cells``) that ``collect_outputs`` reaches, read from the exported class: the formula of a
    Cell (``_f_<name>``, or the method of an uncached Cell) calls another as ``self.<name>``, transitively. None,
    so that every Cell is reset, when a formula reached uses ``self`` in any other way: bare (passed on), or
    through an attribute starting with ``_`` (``_space``, ``_model``, ``_parent``), by which a Cell of the item
    can be reached dynamically; or when a reached Cell has no formula in the class."""
    methods = {item.name.value: item for item in class_node.body.body if isinstance(item, cst.FunctionDef)}
    seen, todo = set(), ['collect_outputs']
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        formula = methods.get(f'_f_{name}', methods.get(name))
        if formula is None:
            return None
        bare, owned = 0, []
        for node in walk(formula.body):
            if isinstance(node, cst.Name) and node.value == 'self':
                bare += 1
            elif isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name) and node.value.value == 'self':
                owned.append(node.attr.value)
        if bare != len(owned) or any(attr.startswith('_') for attr in owned):
            return None
        todo.extend(attr for attr in owned if attr in cells)
    return seen


def class_node(module, cls_name):
    return next(node for node in module.body if isinstance(node, cst.ClassDef) and node.name.value == cls_name)


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


def reset_function(cls_name, block, classes, values, kept=None, reached=None):
    """The ``cdef inline`` function, for the class's .pxd, that returns an item to the state of a new object.

    Every attribute that the class block and its bases declare (the object's layout) is listed: an object
    or a memoryview to None, as ``tp_new`` leaves it; any C value, array or struct to zero bytes, as the allocator leaves
    it. The dense value arrays (``values``) are left unwritten behind their flags, as the class's allocator
    leaves them in a new object (``value_storage``: it zeroes every byte but those arrays). A declaration of another kind stops the build naming the attribute and its Cell.

    For a Cell that ``collect_outputs`` cannot reach (``reached``, from ``reached_cells``; None: every Cell may
    be reached) the item's caches are left as they are: never filled on a reused item, they stay as new.

    The References of ``kept`` (``kept_refs``) keep their values, and ``refs_held`` tells whether each is still the
    one the Space holds (identity, or a C value's bytes); when one is not, the hook copies them all again."""
    structs = _declared(r'^cdef struct (\w+)')
    objects = _OBJECTS | set(classes) | _declared(r'^cdef class (\w+)')
    lines = []
    fields = [*_base_fields(), *_fields(block)]
    types = {name: ctype for _, ctype, name in fields}
    # an enum Reference is held as _mx_definition_<name>, which its property <name> returns
    held_as = lambda name: name if name in types or f'_mx_definition_{name}' not in types else f'_mx_definition_{name}'
    kept = None if kept is None else tuple((held_as(t), held_as(a)) for t, a in kept)
    known = kept is not None and all(t in types and a in types and '[' not in types[t] for t, a in kept)
    held = [] if known else ['False']     # an undeclared or array attribute: copied again at every coordinate
    for target, source in kept if known else ():
        is_object = types[target].split('[')[0].split('.')[-1].strip() in objects or types[target].startswith('_c_')
        held.append(f'root.{target} is base.{source}' if is_object else      # a C value: its bytes (-0.0 is not 0.0)
                    f'memcmp(&root.{target}, &base.{source}, sizeof(root.{target})) == 0')
    for line, ctype, name in fields:
        base = ctype.split('[')[0].split('.')[-1].strip()
        if line in values or known and name in dict(kept):
            continue
        cell = name.removeprefix('_v_').removeprefix('_has_').removeprefix('_spill_')
        if reached is not None and cell != name and cell in classes[cls_name].cells and cell not in reached:
            continue
        if ('[' not in ctype and (base in objects or base.startswith('_c_'))
                or re.fullmatch(r'(const )?[\w ]+\[:(, *:)*\]', ctype)):     # an object or a memoryview
            lines.append(f'    root.{name} = None\n')
        elif (base in _SCALARS or base in structs) and re.fullmatch(r'[\w. ]+(\[\d+\])*', ctype):
            lines.append(f'    memset(&root.{name}, 0, sizeof(root.{name}))\n')
        else:
            cell = name.removeprefix('_v_').removeprefix('_has_').removeprefix('_spill_')
            raise ValueError(f'{cls_name[3:]}: collect cannot reset {line!r} (Cell {cell}) between items')
    source = ('from libc.string cimport memcmp, memset\nfrom cpython.ref cimport Py_REFCNT as _mx_refcount\n\n'
              f'cdef inline void {reset_root(cls_name)}({cls_name} root) noexcept:\n' + ''.join(lines))
    if kept is not None:      # whether the kept References are still the Space's: identity, or a C value's bytes
        source += (f'\n\ncdef inline bint {refs_held(cls_name)}({cls_name} root, {cls_name} base) noexcept:\n'
                   f'    return {" and ".join(held) or "True"}\n')
    return source


def evaluator(workers):
    """The module-level function that every collect hook of a module calls."""
    return cst.parse_module(f'''
def _mx_collect_evaluate(space, coords, outputs_at, refs, reads_parents):
    from ._mx_collect import Accumulator, prepare_output
    import sys
    live = (_mx_sys.BaseMxObject,)
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
    total = [None]      # the Accumulator's own sum dict, once its first output holds only floats in 'sum'
    if workers == 1 or len(coords) == 1 or getattr(sys, '_is_gil_enabled', lambda: True)():
        spare = [None]      # the worker's slot
        for c in coords:
            out = outputs_at(spare, c)
            raw = _mx_collect_plain(out)
            _mx_collect_add(accumulator, total, c, out if raw else prepare_output(out, live), raw, live, prepare_output)
        return accumulator.result()
    import threading
    lock = threading.Lock()
    pending = {{}}
    state = [0, 0, len(coords), None]       # next to take, next to add, stop, the first error

    def work():
        spare = [None]
        while True:
            with lock:
                i = state[0]
                if i >= state[2]:
                    return
                state[0] = i + 1
            try:
                out = outputs_at(spare, coords[i])
                raw = _mx_collect_plain(out)
                out = (out, True) if raw else (prepare_output(out, live), False)
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
                        _mx_collect_add(accumulator, total, coords[k], *pending.pop(k), live, prepare_output)
                    except BaseException as error:
                        state[2], state[3] = k, error
                        break
                    state[1] = k + 1

    threads = [threading.Thread(target=work) for _ in range(min(workers, len(coords)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if state[3] is not None:
        raise state[3]
    return accumulator.result()


def _mx_collect_plain(out):
    # an output that is only a 'sum' of exact floats, added by _mx_collect_add without being prepared
    sums = out.get('sum') if type(out) is dict and len(out) == 1 else None
    return type(sums) is dict and bool(sums) and all(type(v) is float for v in sums.values())


def _mx_collect_add(accumulator, total, c, out, raw, live, prepare_output):
    # out raw (plain, not prepared) or prepared. Raw, with the keys the Accumulator's total holds and floats
    # there: total = total + value, in domain order, as its _add adds; anything else through the Accumulator
    sums, held = out['sum'] if raw else None, total[0]
    if raw and held is not None and len(sums) == len(held) and all(type(held.get(k)) is float for k in sums):
        for k, v in sums.items():
            held[k] = held[k] + v
        return
    accumulator.add(c, prepare_output(out, live) if raw else out)
    held = getattr(accumulator, '_sum', None)
    if (total[0] is None and getattr(accumulator, '_modes', None) == ('sum',) and type(held) is dict
            and all(type(v) is float for v in held.values())):
        total[0] = held
''').body
