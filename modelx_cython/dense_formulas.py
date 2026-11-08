"""Bind proven numeric cache misses without a formula-length eligibility rule."""

import libcst as cst

from .cell_entries import EntryModule, exact_receiver
from .consts import CY_MOD, MX_SELF
from .scopes import CstFacts, OPAQUE_SCOPES
from .syntax import walk


def _eligible(method, keys, guard_safe):
    if not guard_safe:
        return False
    params = method.params
    if (params.posonly_params or params.kwonly_params or params.star_kwarg
            or isinstance(params.star_arg, cst.Param)
            or tuple(param.name.value for param in params.params) != (MX_SELF, *keys)):
        return False
    body = [node for statement in method.body.body for node in walk(statement)]
    facts = CstFacts(method.body.body)
    code = cst.Module([]).code_for_node
    native = {f'{CY_MOD}.{name}' for name in ('double', 'longlong', 'bint')}
    if (method.returns is None or code(method.returns.annotation) not in native
            or any(param.annotation is None or code(param.annotation.annotation) not in native
                   for param in params.params[1:])):
        return False
    if not any(isinstance(item.decorator, cst.Attribute)
               and isinstance(item.decorator.value, cst.Name)
               and item.decorator.value.value == CY_MOD and item.decorator.attr.value == "cfunc"
               for item in method.decorators):
        return False
    if MX_SELF in facts.bindings(body):
        return False
    for node in body:
        if isinstance(node, (*OPAQUE_SCOPES, cst.Yield)):
            return False
        if isinstance(node, cst.Attribute) or isinstance(node, cst.Name) and node in facts.contexts:
            name = node.value if isinstance(node, cst.Name) else node.attr.value
            if name.startswith("__") or name in {"super", "eval", "exec", "locals", "vars"}:
                return False
    return True


class _ExactTests(cst.CSTTransformer):
    def __init__(self, owner, replacement):
        self.test = cst.parse_expression(exact_receiver(owner))
        self.replacement = cst.parse_expression(replacement)

    def leave_Comparison(self, original, updated):
        if original.deep_equals(self.test):
            return self.replacement.with_changes(lpar=(*updated.lpar, *self.replacement.lpar),
                                                 rpar=(*self.replacement.rpar, *updated.rpar))
        return updated


class _Miss(cst.CSTTransformer):
    def __init__(self, formula, direct, keys, propagate=False):
        self.formula = formula
        self.direct = direct
        self.keys = keys
        self.propagate = propagate

    def leave_Call(self, original, updated):
        target = updated.func
        if not (isinstance(target, cst.Attribute) and target.attr.value == self.formula
                and isinstance(target.value, cst.Name) and target.value.value == MX_SELF
                and len(updated.args) == len(self.keys)
                and all(not arg.keyword and not arg.star and isinstance(arg.value, cst.Name)
                        and arg.value.value == key for arg, key in zip(updated.args, self.keys))):
            return updated
        args = (cst.Arg(cst.Name(MX_SELF)), *updated.args)
        if self.propagate:
            args += (cst.Arg(cst.Name("True")),)
        # Only proven exact-receiver calls enter a private cache function.
        # Public cache methods keep their original virtual formula call.
        return updated.with_changes(func=cst.Name(self.direct), args=args)


def bind_entries(state: EntryModule, entries, propagate=False):
    module = state.syntax
    names = state.names
    exact = names.reserve('_mx_exact_receiver')
    receiver_types = {fn.name.value: fn.params.params[0].annotation
                      for fn in module.body if isinstance(fn, cst.FunctionDef)
                      and fn.name.value in entries}
    targets = {(layout.owner, layout.formula): (entry, layout)
               for entry, layout in entries.items() if entry in receiver_types}
    replacements, methods, misses = [], {}, {}
    for cls in module.body:
        if not isinstance(cls, cst.ClassDef) or not isinstance(cls.body, cst.IndentedBlock):
            continue
        for fn in cls.body.body:
            if not isinstance(fn, cst.FunctionDef):
                continue
            identity = cls.name.value, fn.name.value
            if identity not in targets:
                continue
            entry, layout = targets[identity]
            keys = layout.keys
            if not _eligible(fn, keys, layout.guard_safe):
                continue
            direct = names.reserve(f'_mx_dense_formula_{len(replacements)}')
            receiver, *params = fn.params.params
            receiver = receiver.with_changes(annotation=receiver_types[entry])
            args = ", ".join((MX_SELF, *keys))
            formula_body = fn.body
            if propagate:
                params = [param.with_changes(default=None, equal=cst.MaybeSentinel.DEFAULT)
                          for param in params]
                params.append(cst.Param(cst.Name(exact), annotation=cst.Annotation(
                    cst.parse_expression(f"{CY_MOD}.bint"))))
                formula_body = formula_body.visit(_ExactTests(layout.owner, exact))
                args += ", " + exact_receiver(layout.owner)
            replacements.append(fn.with_changes(
                name=cst.Name(direct),
                body=formula_body,
                params=fn.params.with_changes(params=(receiver, *params)),
                decorators=(*fn.decorators, cst.Decorator(cst.parse_expression(f"{CY_MOD}.inline"))),
            ))
            methods[fn] = fn.with_changes(body=fn.body.with_changes(body=(
                cst.parse_statement(f"return {direct}({args})\n"),
            )))
            misses[entry] = _Miss(layout.formula, direct, keys, propagate)
    if not replacements:
        return state
    body = []
    for stmt in module.body:
        if isinstance(stmt, cst.ClassDef) and isinstance(stmt.body, cst.IndentedBlock):
            stmt = stmt.with_changes(body=stmt.body.with_changes(
                body=tuple(methods.get(item, item) for item in stmt.body.body)))
        elif isinstance(stmt, cst.FunctionDef) and stmt.name.value in misses:
            stmt = stmt.with_changes(body=stmt.body.visit(misses[stmt.name.value]))
        body.append(stmt)
    return state.replace(module.with_changes(body=(*body, *replacements)))
