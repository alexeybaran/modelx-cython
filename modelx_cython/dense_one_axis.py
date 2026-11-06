"""Separate checked and proven calls to a single dense cache body."""

import ast

import libcst as cst

from .cell_entries import dense_condition
from .consts import CY_MOD
from .dense_formulas import _ExactTests


class _Calls(cst.CSTTransformer):
    def __init__(self, entries, propagate):
        self.entries = entries
        self.propagate = propagate

    def leave_Call(self, original, updated):
        if not isinstance(original.func, cst.Name) or original.func.value not in self.entries:
            return updated
        if (len(original.args) != 3 or not isinstance(original.args[-1].value, cst.Integer)
                or original.args[-1].value.evaluated_value not in (0, 1)):
            raise ValueError("unexpected one-axis proof call")
        target = self.entries[original.func.value] if original.args[-1].value.evaluated_value else updated.func
        args = (*updated.args[:-2], updated.args[-2].with_changes(comma=cst.MaybeSentinel.DEFAULT))
        if self.propagate and original.args[-1].value.evaluated_value:
            args += (cst.Arg(cst.Name("True")),)
        return updated.with_changes(func=target, args=args)


def _rewrite(source, layouts, propagate=False):
    module = cst.parse_module(source)
    names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
    names.update(stmt.name.value for stmt in module.body if isinstance(stmt, cst.FunctionDef))
    exact = "_mx_inbounds_exact"
    while exact in names:
        exact += "_"
    bodies, rewritten, proved = [], [], {}
    for stmt in module.body:
        if not isinstance(stmt, cst.FunctionDef) or stmt.name.value not in layouts:
            rewritten.append(stmt)
            continue
        layout = layouts[stmt.name.value]
        params = stmt.params.params
        if (len(layout.axes) != 1 or layout.exact_spill or layout.axes[0].c_type != "long long"
                or not isinstance(stmt.body, cst.IndentedBlock) or len(stmt.body.body) != 1):
            rewritten.append(stmt)
            continue
        axis, = layout.axes
        guard, = stmt.body.body
        proof = params[-1].name.value if len(params) == 3 else None
        expected = ast.parse(dense_condition(layout.axes, proof), mode="eval").body
        if (not isinstance(guard, cst.If) or not isinstance(guard.orelse, cst.Else)
                or ast.dump(ast.parse(module.code_for_node(guard.test), mode="eval").body) != ast.dump(expected)):
            rewritten.append(stmt)
            continue
        span = axis.stop - axis.lower
        if not 0 < span < 1 << 64:
            rewritten.append(stmt)
            continue
        cast = f"{CY_MOD}.cast"
        unsigned = f"{CY_MOD}.ulonglong"
        check = cst.parse_expression(
            f"({cast}({unsigned}, {axis.key}) - {cast}({unsigned}, {axis.lower})) < {span}")
        if proof is None:
            rewritten.append(stmt.with_changes(body=stmt.body.with_changes(
                body=(guard.with_changes(test=check),))))
            continue
        name = stmt.name.value + "_inbounds"
        while name in names:
            name += "_"
        names.add(name)
        proved[stmt.name.value] = cst.Name(name)
        args = tuple(param.with_changes(comma=cst.MaybeSentinel.DEFAULT) for param in params[:-1])
        signature = stmt.params.with_changes(params=args)
        body_signature = signature
        body = guard.body
        flag = ""
        if propagate:
            body_signature = signature.with_changes(params=(*args, cst.Param(
                cst.Name(exact), annotation=cst.Annotation(cst.parse_expression(f"{CY_MOD}.bint")))))
            body = body.visit(_ExactTests(layout.owner, f"({exact} or type(self) is {layout.owner})"))
            flag = ", False"
        bodies.append(stmt.with_changes(name=cst.Name(name), params=body_signature, body=body))
        entry = cst.parse_statement(f"return {name}({args[0].name.value}, {axis.key}{flag})\n")
        rewritten.append(stmt.with_changes(params=signature, body=stmt.body.with_changes(
            body=(guard.with_changes(test=check, body=guard.body.with_changes(body=(entry,))),))))
    return module.with_changes(body=(*rewritten, *bodies)).visit(_Calls(proved, propagate)).bytes


def specialize_one_axis(sources, entries, propagate=False):
    for name, layouts in entries.items():
        path = sources[name]
        original = path.read_bytes()
        rewritten = _rewrite(original, layouts, propagate)
        if rewritten != original:
            path.write_bytes(rewritten)
