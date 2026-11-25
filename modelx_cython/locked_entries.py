"""Avoid re-entering a generated dense Cell's RLock inside the same lock."""
from dataclasses import dataclass

import libcst as cst

from .cell_entries import exact_receiver
from .consts import CY_MOD, FORMULA_PREF, MX_LOCK, MX_SELF
from .inline_entries import allows_exact_receiver
from .scopes import Names, OPAQUE_SCOPES
from .syntax import dotted_name, walk


@dataclass(frozen=True)
class Entry:
    wrapper: cst.FunctionDef
    formula: cst.FunctionDef
    lock: cst.With
    keys: tuple[str, ...]


def _entry(state, cls, cell):
    if (not cell.has_formula_def or not cell.is_locked or not cell.has_args() or not cell.has_typeinfo()
            or not cell.is_arrayable() or cell.has_dense_spill
            or cell.called_with_kwargs or cell.body_has_closure):
        return None
    if not cell.params or any(cell.get_argtype_expr(p, c_style=True) != 'long long' for p in cell.params):
        return None
    wrapper = state.methods.get((cls.name, cell.name))
    formula = state.methods.get((cls.name, FORMULA_PREF + cell.name))
    if (wrapper is None or formula is None or formula.asynchronous
            or not isinstance(wrapper.body, cst.IndentedBlock) or not allows_exact_receiver(wrapper, cls.name)):
        return None
    params = formula.params
    if (params.posonly_params or params.kwonly_params or isinstance(params.star_arg, cst.Param)
            or params.star_kwarg or any(param.default is not None for param in params.params)
            or tuple(p.name.value for p in params.params) != (MX_SELF, *cell.params)):
        return None
    if len(wrapper.body.body) != 1 or not isinstance((guard := wrapper.body.body[0]), cst.If):
        return None
    statements = guard.body.body
    if len(statements) != 2 or not isinstance((lock := statements[1]), cst.With):
        return None
    if len(lock.items) != 1 or dotted_name(lock.items[0].item) != f'{MX_SELF}.{MX_LOCK}':
        return None
    names = Names(formula).names
    if (any(isinstance(node, (*OPAQUE_SCOPES, cst.Yield, cst.Await)) for node in walk(formula.body))
            or {'super', '__class__', 'locals', 'globals', 'eval', 'exec'} & names
            or any(name.startswith('__') and not name.endswith('__') for name in names)):
        return None
    return Entry(wrapper, formula, lock, tuple(cell.params))


def _same_receiver_calls(formula, entries):
    scopes = cst.MetadataWrapper(cst.Module((formula,)), unsafe_skip_copy=True).resolve(cst.metadata.ScopeProvider)
    receiver = formula.params.params[0]
    calls = {}
    for node in walk(formula.body):
        if not isinstance(node, cst.Call) or not isinstance(node.func, cst.Attribute):
            continue
        base, target = node.func.value, node.func.attr.value
        if (not isinstance(base, cst.Name) or base.value != MX_SELF or target not in entries
                or len(node.args) != len(entries[target].keys)
                or any(arg.keyword is not None or arg.star for arg in node.args)):
            continue
        scope = scopes.get(base)
        bindings = tuple(scope[MX_SELF]) if scope is not None else ()
        if len(bindings) == 1 and bindings[0].node is receiver:
            calls[node] = target
    return calls


class _Calls(cst.CSTTransformer):
    def __init__(self, calls, targets, *, guard=None):
        self.calls = calls
        self.targets = targets
        self.guard = guard

    def leave_Call(self, old, new):
        if (name := self.calls.get(old)) is None:
            return new
        call = new.with_changes(func=cst.Name(self.targets[name]), args=(cst.Arg(cst.Name(MX_SELF)), *new.args))
        if self.guard is not None:
            return cst.IfExp(self.guard, call.with_changes(lpar=(), rpar=()), new.with_changes(lpar=(), rpar=()),
                             lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))
        return call


def _as_function(node, owner, name, body):
    receiver, *params = node.params.params
    receiver = receiver.with_changes(annotation=cst.Annotation(cst.Name(owner)))
    return node.with_changes(name=cst.Name(name), body=body,
                             params=node.params.with_changes(params=(receiver, *params)),
                             decorators=(cst.Decorator(cst.parse_expression(f'{CY_MOD}.cfunc')),))


def lower_locked_entries(state, module_info):
    """Fork the native miss body only for an actual lexical lock and exact receiver."""
    replacements = {}
    helpers = []
    origins = dict(state.function_origins)
    for cls in module_info.classes.values():
        entries = {cell.name: entry for cell in cls.cells.values() if (entry := _entry(state, cls, cell))}
        calls = {name: found for name, entry in entries.items()
                 if (found := _same_receiver_calls(entry.formula, entries))}
        if not calls:
            continue
        targets = {target for sites in calls.values() for target in sites.values()}
        locked = {target: state.names.reserve(f'_mx_held_cache_{cls.name}_{target}') for target in sorted(targets)}
        formulas = {name: state.names.reserve(f'_mx_held_formula_{cls.name}_{name}') for name in calls}
        guard = cst.parse_expression(exact_receiver(cls.name))
        for name, sites in calls.items():
            entry = entries[name]
            cloned = entry.formula.body.visit(_Calls(sites, locked))
            helpers.append(_as_function(entry.formula, cls.name, formulas[name], cloned))
            origins[formulas[name]] = (cls.name, entry.formula.name.value)
            locked_body = entry.lock.body
            formula_calls = {node: name for node in walk(locked_body)
                             if isinstance(node, cst.Call)
                             and dotted_name(node.func) == f'{MX_SELF}.{FORMULA_PREF}{name}'}
            assert len(formula_calls) == 1
            patched = entry.lock.with_changes(body=locked_body.visit(_Calls(formula_calls, formulas, guard=guard)))
            branch = entry.wrapper.body.body[0]
            body = branch.body.with_changes(body=(branch.body.body[0], patched))
            replacements[entry.wrapper] = entry.wrapper.with_changes(
                body=entry.wrapper.body.with_changes(body=(branch.with_changes(body=body),)))
        for name, helper in locked.items():
            entry = entries[name]
            branch = entry.wrapper.body.body[0]
            locked_body = entry.lock.body
            if name in formulas:
                formula_calls = {node: name for node in walk(locked_body)
                                 if isinstance(node, cst.Call)
                                 and dotted_name(node.func) == f'{MX_SELF}.{FORMULA_PREF}{name}'}
                locked_body = locked_body.visit(_Calls(formula_calls, formulas))
            # All accesses here occur under the caller's lock. The original
            # publication and out-of-bounds branch still apply.
            branch = branch.with_changes(body=locked_body)
            helpers.append(_as_function(entry.wrapper, cls.name, helper,
                                        entry.wrapper.body.with_changes(body=(branch,))))
            origins[helper] = (cls.name, entry.formula.name.value)
    if not helpers:
        return state
    body = tuple(node.with_changes(body=node.body.with_changes(
        body=tuple(replacements.get(item, item) for item in node.body.body)))
        if isinstance(node, cst.ClassDef) else node for node in state.syntax.body)
    return state.replace(state.syntax.with_changes(body=(*body, *helpers)), origins)
