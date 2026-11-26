"""Compiled ``collect()``: the hook of modelx's exported collect, version 1.

A Space that declares collect is exported with ``_mx_collect_item`` (the ItemSpace at one coordinate:
an existing one, or a new root that is not stored) and a method that calls
``_mx_sys.collect_from_space`` (``_mx_collect_run``, which the Space's ``collect`` property returns;
an ItemSpace has no ``collect``, as in live modelx). The compiled class adds ``_mx_collect_hook``, which modelx calls
with the checked domain: each item is built (as ``__call__`` builds a new root, by ``item_roots``, for
a Space without child Spaces; else by ``_mx_collect_item``), its ``collect_outputs()`` read and checked
by the export's ``prepare_output``, and the item released when collect created it. The outputs are added
by the export's ``Accumulator`` in domain order, so the result is the one live modelx computes, bit
for bit, at any number of workers.

Workers: the spec key ``collect_workers`` (default 1), or ``_mx_collect_workers`` set on the compiled
model object. With more than one, items run on that many threads; their outputs are added in domain
order as they complete, and an error raises the exception of the first failing coordinate, as one
worker does. Threads run model code at the same time only on free-threaded Python; with the GIL,
one worker runs. Before starting threads, a Reference of the Space that is a Space without a lock
(not exported in ``locked_spaces``) raises ValueError naming it, as does a formula of the Space that
reads ``_model``, ``_parent`` or ``_space``: what those reach cannot be filled by two threads at once.
"""
import libcst as cst

from . import item_roots
from .syntax import dotted_name

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


def hook(class_node, cls_name, refs, fast=True):
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
    outputs_at = lam.with_changes(body=lam.body.with_changes(func=cst.Name('_mx_outputs')))
    unpack = f'    {_code(others)} = args\n' if others.elements else ''
    apply = ('            _mx_fast._mx_apply_formula(%s)\n' % params
             if '_mx_apply_formula' in _code(item) else '')
    built = (
        f'        if type(self) is {cls_name} and not self._mx_spaces and not self._mx_roots:\n'
        f'            _mx_fast: {cls_name} = {item_roots.new_root(cls_name)}(self)\n'
        f'            self._mx_assign_params(_mx_fast, {params})\n'
        f'{apply}'
        f'            _mx_out = _mx_fast.collect_outputs()\n'
        f'            {item_roots.release_root(cls_name)}(_mx_fast)\n'
        f'            return _mx_out\n') if fast else ''
    source = (
        f'_mx_collect_hook_version = {HOOK_VERSION}\n'
        f'\n'
        f'def _mx_collect_hook(self, coords, args):\n'
        f'{unpack}'
        f'    def _mx_outputs({params}):\n'
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
    if workers == 1 or len(coords) == 1 or getattr(sys, '_is_gil_enabled', lambda: True)():
        for c in coords:
            accumulator.add(c, prepare_output(outputs_at(c), live))
        return accumulator.result()
    name = type(space).__name__[len('_c_'):]
    if reads_parents:
        raise ValueError(f'collect of {{name}} with {{workers}} workers: a formula of {{name}} reads _model, '
                         '_parent or _space, which the workers could fill at once; use one worker')
    for ref in refs:
        value = getattr(space, ref, None)
        if isinstance(value, _mx_sys.BaseSpace) and value is not space and value._mx_lock is None:
            raise ValueError(f'collect of {{name}} with {{workers}} workers: the Reference {{ref}} is a Space '
                             'without a lock; export it in locked_spaces or use one worker')
    import threading
    lock = threading.Lock()
    pending = {{}}
    state = [0, 0, len(coords), None]       # next to take, next to add, stop, the first error

    def work():
        while True:
            with lock:
                i = state[0]
                if i >= state[2]:
                    return
                state[0] = i + 1
            try:
                out = prepare_output(outputs_at(coords[i]), live)
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

    threads = [threading.Thread(target=work) for _ in range(min(workers, len(coords)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if state[3] is not None:
        raise state[3]
    return accumulator.result()
''').body
