"""Bind small dense cache misses before handing source to Cython."""

import ast

import libcst as cst

from .cell_entries import exact_receiver
from .consts import CY_MOD, MX_SELF


def _eligible(method, syntax, keys):
    params = method.params
    if (params.posonly_params or params.kwonly_params or params.star_kwarg
            or isinstance(params.star_arg, cst.Param)
            or tuple(param.name.value for param in params.params) != (MX_SELF, *keys)):
        return False
    body = [node for statement in syntax.body for node in ast.walk(statement)]
    if len(body) > 150:
        return False
    if not any(isinstance(item.decorator, cst.Attribute)
               and isinstance(item.decorator.value, cst.Name)
               and item.decorator.value.value == CY_MOD and item.decorator.attr.value == "cfunc"
               for item in method.decorators):
        return False
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Lambda, ast.Yield, ast.YieldFrom, ast.GeneratorExp,
                             ast.ListComp, ast.SetComp, ast.DictComp)):
            return False
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
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
    def __init__(self, owner, formula, direct, keys, propagate=False):
        self.guard = cst.parse_expression(exact_receiver(owner))
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
        direct = updated.with_changes(func=cst.Name(self.direct), args=args)
        return cst.IfExp(test=self.guard, body=direct, orelse=updated,
                         lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))


def _rewrite(source, entries, propagate=False):
    tree = ast.parse(source)
    module = cst.parse_module(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names.update(node.name for node in ast.walk(tree)
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)))
    names.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    names.update(node.asname or node.name.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.alias))
    exact = "_mx_exact_receiver"
    while exact in names:
        exact += "_"
    receiver_types = {fn.name.value: fn.params.params[0].annotation
                      for fn in module.body if isinstance(fn, cst.FunctionDef)
                      and fn.name.value in entries}
    targets = {(layout.owner, layout.formula): (entry, layout)
               for entry, layout in entries.items() if entry in receiver_types}
    formulas = {(cls.name, node.name): node for cls in tree.body if isinstance(cls, ast.ClassDef)
                for node in cls.body if isinstance(node, ast.FunctionDef)}
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
            keys = tuple(axis.key for axis in layout.axes)
            if not _eligible(fn, formulas[identity], keys):
                continue
            if propagate and any(isinstance(node, ast.Name) and node.id == MX_SELF
                                 and not isinstance(node.ctx, ast.Load)
                                 for node in ast.walk(formulas[identity])):
                continue
            direct = f"_mx_dense_formula_{len(replacements)}"
            while direct in names:
                direct += "_"
            names.add(direct)
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
            misses[entry] = _Miss(layout.owner, layout.formula, direct, keys, propagate)
    if not replacements:
        return source
    body = []
    for stmt in module.body:
        if isinstance(stmt, cst.ClassDef) and isinstance(stmt.body, cst.IndentedBlock):
            stmt = stmt.with_changes(body=stmt.body.with_changes(
                body=tuple(methods.get(item, item) for item in stmt.body.body)))
        elif isinstance(stmt, cst.FunctionDef) and stmt.name.value in misses:
            stmt = stmt.with_changes(body=stmt.body.visit(misses[stmt.name.value]))
        body.append(stmt)
    return module.with_changes(body=(*body, *replacements)).bytes


def bind_dense_formulas(modules, layouts, propagate=False):
    for module, entries in layouts.items():
        path = modules[module]
        original = path.read_bytes()
        rewritten = _rewrite(original, entries, propagate)
        if rewritten != original:
            path.write_bytes(rewritten)
