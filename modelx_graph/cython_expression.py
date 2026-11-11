from __future__ import annotations

"""Shared native Cython expression lowering.

The Stage/compact and recursive backends deliberately share this expression
converter.  Execution topology is supplied by ``emitter._cell_call`` and input
ABI details by ``emitter._input_expr``; scalar Python syntax, normalized lookup
intrinsics and math calls are lowered once here.
"""

import ast
import math

from .codegen_common import CodegenError


def _module_name(emitter, receiver: str) -> str | None:
    explicit = getattr(emitter, "module_refs", None)
    if explicit and receiver in explicit:
        return explicit[receiver]
    model = getattr(emitter, "m", None)
    space = getattr(model, "space", None)
    refs = getattr(space, "refs", {}) if space is not None else {}
    obj = refs.get(receiver) if hasattr(refs, "get") else None
    name = getattr(obj, "__name__", None)
    if name is not None:
        return str(name)
    if receiver == "math":
        return "math"
    if receiver in {"np", "numpy"}:
        return "numpy"
    return None


def emit_cython_expr(emitter, n: ast.AST, ctx) -> str:
    """Lower one already-normalized scalar expression to Cython source.

    ``emitter`` must provide ``expr`` (normally delegating back here), ``v``,
    ``input_specs``, ``_cell_call`` and ``_input_expr``.  This intentionally has
    no Stage/storage dependency.
    """
    overrides = ctx.get("expr_overrides")
    if overrides:
        key = ast.dump(n, include_attributes=False)
        if key in overrides:
            return overrides[key]
    if isinstance(n, ast.Constant):
        if n.value is None:
            raise CodegenError("None is not a native numeric value")
        if isinstance(n.value, float) and math.isinf(n.value):
            return "INFINITY" if n.value > 0 else "-INFINITY"
        return repr(n.value)
    if isinstance(n, ast.Name):
        if n.id in ctx.get("locals", {}):
            return ctx["locals"][n.id]
        if n.id == ctx.get("loop_var"):
            return ctx.get("loop_var_c", n.id)
        if n.id in ("True", "False"):
            return n.id
        raise CodegenError(f"unresolved name {n.id}")
    if isinstance(n, ast.UnaryOp):
        op = {ast.USub: "-", ast.UAdd: "+", ast.Not: "not "}.get(type(n.op))
        if op is None:
            raise CodegenError(f"unsupported unary {type(n.op).__name__}")
        return f"({op}{emitter.expr(n.operand, ctx)})"
    if isinstance(n, ast.BinOp):
        a = emitter.expr(n.left, ctx)
        b = emitter.expr(n.right, ctx)
        if isinstance(n.op, ast.Pow):
            return f"pow(({a}), ({b}))"
        if isinstance(n.op, ast.Div):
            return f"((<double>({a})) / ({b}))"
        op = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.FloorDiv: "//",
            ast.Mod: "%",
        }.get(type(n.op))
        if op is None:
            raise CodegenError(f"unsupported binary {type(n.op).__name__}")
        return f"(({a}) {op} ({b}))"
    if isinstance(n, ast.BoolOp):
        op = " and " if isinstance(n.op, ast.And) else " or " if isinstance(n.op, ast.Or) else None
        if op is None:
            raise CodegenError("unsupported boolean operator")
        return "(" + op.join(emitter.expr(x, ctx) for x in n.values) + ")"
    if isinstance(n, ast.Compare):
        if (
            len(n.ops) == 1
            and isinstance(n.ops[0], (ast.In, ast.NotIn))
            and len(n.comparators) == 1
            and isinstance(n.comparators[0], (ast.Tuple, ast.List, ast.Set))
        ):
            values = []
            for elt in n.comparators[0].elts:
                if not (
                    isinstance(elt, ast.Constant)
                    and isinstance(elt.value, (bool, int, float))
                    and elt.value is not None
                ):
                    raise CodegenError(
                        f"comparison {type(n.ops[0]).__name__} requires a finite native numeric literal set"
                    )
                values.append(elt)
            left = emitter.expr(n.left, ctx)
            if not values:
                return "True" if isinstance(n.ops[0], ast.NotIn) else "False"
            checks = [f"(({left}) == ({emitter.expr(elt, ctx)}))" for elt in values]
            member = "(" + " or ".join(checks) + ")"
            return f"(not {member})" if isinstance(n.ops[0], ast.NotIn) else member
        ops = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Is: "is",
            ast.IsNot: "is not",
        }
        parts = [emitter.expr(n.left, ctx)]
        for op, comp in zip(n.ops, n.comparators):
            if type(op) not in ops or isinstance(op, (ast.Is, ast.IsNot)):
                raise CodegenError(f"comparison {type(op).__name__} requires Python fallback")
            parts += [ops[type(op)], emitter.expr(comp, ctx)]
        return "(" + " ".join(parts) + ")"
    if isinstance(n, ast.IfExp):
        return f"({emitter.expr(n.body, ctx)} if {emitter.expr(n.test, ctx)} else {emitter.expr(n.orelse, ctx)})"
    if isinstance(n, ast.Call):
        if isinstance(n.func, ast.Name):
            name = n.func.id
            if name in emitter.v:
                return emitter._cell_call(name, n.args, ctx)
            if name in ("point_input", "global_input", "array_input", "table_input"):
                return emitter._input_expr(name, n.args, ctx)
            if name in ("max", "min"):
                if len(n.args) < 1:
                    raise CodegenError(f"{name} requires arguments")
                xs = [emitter.expr(x, ctx) for x in n.args]
                cur = xs[0]
                for x in xs[1:]:
                    cmp = ">=" if name == "max" else "<="
                    cur = f"(({cur}) if ({cur}) {cmp} ({x}) else ({x}))"
                return cur
            if name == "abs" and len(n.args) == 1:
                return f"fabs({emitter.expr(n.args[0], ctx)})"
            if name == "round" and 1 <= len(n.args) <= 2:
                x = emitter.expr(n.args[0], ctx)
                if len(n.args) == 1:
                    return f"rint({x})"
                if isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, int):
                    scale = 10.0 ** n.args[1].value
                    return f"(rint(({x}) * {scale!r}) / {scale!r})"
                digits_dtype = expression_dtype(emitter, n.args[1], ctx.get("local_dtypes", ctx.get("locals", {})))
                if digits_dtype not in {"bool", "int64"}:
                    raise CodegenError("round ndigits must be integral")
                digits = emitter.expr(n.args[1], ctx)
                # Fixed-arity scalar lowering.  Do not thread model inputs through
                # a helper merely because ndigits is dynamic; pow/rint stay local.
                scale = f"pow(10.0, <double>({digits}))"
                return f"(rint(({x}) * ({scale})) / ({scale}))"
            if name == "int" and len(n.args) == 1:
                return f"(<long>({emitter.expr(n.args[0], ctx)}))"
            if name == "float" and len(n.args) == 1:
                return f"(<double>({emitter.expr(n.args[0], ctx)}))"
            if name == "bool" and len(n.args) == 1:
                return f"(<bint>({emitter.expr(n.args[0], ctx)}))"
            if name == "__is_missing__" and len(n.args) == 1:
                return f"isnan(<double>({emitter.expr(n.args[0], ctx)}))"
            if name == "__is_inf__" and len(n.args) == 1:
                return f"isinf(<double>({emitter.expr(n.args[0], ctx)}))"
            if name == "__erf__" and len(n.args) == 1:
                return f"erf(<double>({emitter.expr(n.args[0], ctx)}))"
            if name == "__exact_axis_code__" and len(n.args) == 4:
                value = emitter.expr(n.args[0], ctx)
                start = emitter.expr(n.args[1], ctx)
                step = emitter.expr(n.args[2], ctx)
                # ``size`` is part of the proof contract even though the dense
                # arithmetic itself does not need it at runtime.
                emitter.expr(n.args[3], ctx)
                return f"(<long>((({value}) - ({start})) // ({step})))"
            if name == "__exact_axis_code_valid__" and len(n.args) == 5:
                value = emitter.expr(n.args[0], ctx)
                valid = emitter.expr(n.args[1], ctx)
                start = emitter.expr(n.args[2], ctx)
                step = emitter.expr(n.args[3], ctx)
                size = emitter.expr(n.args[4], ctx)
                code = f"(<long>((({value}) - ({start})) // ({step})))"
                in_range = f"(({code}) >= 0 and ({code}) < ({size}))"
                return (
                    f"(({code}) if (({valid}) and {in_range}) "
                    f"else __exact_axis_invalid__(-93002))"
                )
            if name == "__prepared_point_lookup__" and len(n.args) == 2:
                if not all(
                    isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    for arg in n.args
                ):
                    raise CodegenError("prepared point lookup requires literal internal input keys")
                value_key, valid_key = (arg.value for arg in n.args)
                value_spec = emitter.input_specs.get(value_key)
                valid_spec = emitter.input_specs.get(valid_key)
                if value_spec is None or valid_spec is None:
                    raise CodegenError("prepared point lookup references unknown input")
                if value_spec.scope != "point" or value_spec.ndim != 1:
                    raise CodegenError("prepared point lookup value must be point-scoped 1-D")
                if valid_spec.scope != "point" or valid_spec.ndim != 1 or valid_spec.dtype != "bool":
                    raise CodegenError("prepared point lookup validity must be point-scoped bool")
                value = emitter._input_expr("point_input", [ast.Constant(value_key)], ctx)
                valid = emitter._input_expr("point_input", [ast.Constant(valid_key)], ctx)
                ctype = emitter.ctype(value_spec.dtype)
                return (
                    f"(({value}) if ({valid}) else "
                    f"<{ctype}>__exact_axis_invalid__(-93003))"
                )
            if name == "__sparse_table_lookup__" and len(n.args) >= 3:
                if not (
                    isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, str)
                    and isinstance(n.args[1], ast.Constant)
                    and isinstance(n.args[1].value, str)
                ):
                    raise CodegenError("sparse table lookup requires literal internal input keys")
                keys_key, values_key = n.args[0].value, n.args[1].value
                keys_spec = emitter.input_specs.get(keys_key)
                values_spec = emitter.input_specs.get(values_key)
                if keys_spec is None or values_spec is None:
                    raise CodegenError("sparse table lookup references unknown input")
                ndim = len(n.args) - 2
                if keys_spec.dtype != "int64" or keys_spec.ndim != 2:
                    raise CodegenError("sparse table coordinate input must be a 2-D int64 buffer")
                if values_spec.ndim != 1 or values_spec.dtype not in {"float64", "int64", "bool"}:
                    raise CodegenError("sparse table values must be a 1-D native numeric buffer")
                indices = ", ".join(
                    f"<long>({emitter.expr(arg, ctx)})" for arg in n.args[2:]
                )
                return (
                    f"_sparse_table_lookup_{ndim}_{values_spec.dtype}"
                    f"({keys_key}, {values_key}, {indices})"
                )
            if name == "__step_lookup_1d__" and len(n.args) == 5:
                if not (isinstance(n.args[0], ast.Constant) and isinstance(n.args[1], ast.Constant)):
                    raise CodegenError("step lookup requires literal internal input keys")
                xkey, ykey = n.args[0].value, n.args[1].value
                if xkey not in emitter.input_specs or ykey not in emitter.input_specs:
                    raise CodegenError("step lookup references unknown input")
                return (
                    f"_step_lookup_1d({xkey}, {ykey}, "
                    f"<double>({emitter.expr(n.args[2], ctx)}), <double>({emitter.expr(n.args[3], ctx)}), "
                    f"<long>({emitter.expr(n.args[4], ctx)}))"
                )
            if name == "__interval_lookup_1d__" and len(n.args) in {4, 7}:
                if not all(isinstance(x, ast.Constant) for x in n.args[:3]):
                    raise CodegenError("interval lookup requires literal internal input keys")
                lokey, hikey, ykey = (x.value for x in n.args[:3])
                if any(k not in emitter.input_specs for k in (lokey, hikey, ykey)):
                    raise CodegenError("interval lookup references unknown input")
                if len(n.args) == 4:
                    closure_mode, fallback_mode, default = "0", "0", "0.0"
                else:
                    closure_mode = f"<long>({emitter.expr(n.args[4], ctx)})"
                    fallback_mode = f"<long>({emitter.expr(n.args[5], ctx)})"
                    default = f"<double>({emitter.expr(n.args[6], ctx)})"
                return (
                    f"_interval_lookup_1d({lokey}, {hikey}, {ykey}, "
                    f"<double>({emitter.expr(n.args[3], ctx)}), {closure_mode}, {fallback_mode}, {default})"
                )
            if name == "__interp_lookup_1d__" and len(n.args) == 4:
                if not (
                    isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[1], ast.Constant)
                    and isinstance(n.args[3], ast.Constant)
                ):
                    raise CodegenError("interpolation lookup requires literal internal input keys and mode")
                xkey, ykey = n.args[0].value, n.args[1].value
                if xkey not in emitter.input_specs or ykey not in emitter.input_specs:
                    raise CodegenError("interpolation lookup references unknown input")
                return (
                    f"_interp_lookup_1d({xkey}, {ykey}, "
                    f"<double>({emitter.expr(n.args[2], ctx)}), {int(n.args[3].value)})"
                )
        if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
            receiver = n.func.value.id
            attr = n.func.attr
            module_name = _module_name(emitter, receiver)
            supported = {
                "exp": "exp",
                "log": "log",
                "sqrt": "sqrt",
                "sin": "sin",
                "cos": "cos",
                "floor": "floor",
                "ceil": "ceil",
            }
            if module_name in ("math", "numpy") and attr in supported and len(n.args) == 1:
                return f"{supported[attr]}({emitter.expr(n.args[0], ctx)})"
        raise CodegenError(f"unsupported call: {ast.unparse(n)}")
    raise CodegenError(f"unsupported expression: {ast.dump(n, include_attributes=False)}")


def expression_dtype(emitter, node: ast.AST, locals_: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "bool"
        if isinstance(node.value, int):
            return "int64"
        if isinstance(node.value, float):
            return "float64"
        return None
    if isinstance(node, ast.Name):
        return locals_.get(node.id)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return "bool"
        return expression_dtype(emitter, node.operand, locals_)
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return "bool"
    if isinstance(node, ast.IfExp):
        a = expression_dtype(emitter, node.body, locals_)
        b = expression_dtype(emitter, node.orelse, locals_)
        if a == b:
            return a
        if {a, b} <= {"bool", "int64"}:
            return "int64"
        if a is not None and b is not None:
            return "float64"
        return a or b
    if isinstance(node, ast.BinOp):
        a = expression_dtype(emitter, node.left, locals_)
        b = expression_dtype(emitter, node.right, locals_)
        if isinstance(node.op, ast.Div):
            return "float64"
        if a in ("bool", "int64") and b in ("bool", "int64") and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
        ):
            return "int64"
        if a is not None and b is not None:
            return "float64"
        return a or b
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name in emitter.v:
            return emitter.v[name].dtype
        if name in ("point_input", "global_input", "array_input", "table_input"):
            if node.args and isinstance(node.args[0], ast.Constant):
                spec = emitter.input_specs.get(node.args[0].value)
                return None if spec is None else spec.dtype
        if name in ("__exact_axis_code__", "__exact_axis_code_valid__"):
            return "int64"
        if name == "__prepared_point_lookup__" and len(node.args) == 2:
            if isinstance(node.args[0], ast.Constant):
                spec = emitter.input_specs.get(node.args[0].value)
                return None if spec is None else spec.dtype
        if name == "__sparse_table_lookup__" and len(node.args) >= 2:
            if isinstance(node.args[1], ast.Constant):
                spec = emitter.input_specs.get(node.args[1].value)
                return None if spec is None else spec.dtype
        if name == "int":
            return "int64"
        if name == "bool":
            return "bool"
        if name == "float":
            return "float64"
        if name in ("__is_missing__", "__is_inf__"):
            return "bool"
        if name in ("min", "max", "abs") and node.args:
            kinds = [expression_dtype(emitter, arg, locals_) for arg in node.args]
            if all(k in ("bool", "int64") for k in kinds):
                return "int64"
            if all(k is not None for k in kinds):
                return "float64"
    return None


def infer_local_types(emitter, fn: ast.FunctionDef, parameter_types: dict[str, str] | None = None) -> dict[str, str]:
    assignments: list[tuple[str, ast.AST]] = []
    names: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                    assignments.append((target.id, n.value))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            names.add(n.target.id)
            assignments.append((n.target.id, n.value))
        elif isinstance(n, ast.For) and isinstance(n.target, ast.Name):
            # Production admission remains fail-closed in the backend.  Once a
            # local range loop is admitted, its iterator has Python range's
            # integer semantics and must not default to float64.
            names.add(n.target.id)
    inferred: dict[str, str] = dict(parameter_types or {})
    for n in ast.walk(fn):
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name):
            inferred[n.target.id] = "int64"
    for _ in range(max(1, len(names) + 1)):
        changed = False
        for name, expr in assignments:
            new = expression_dtype(emitter, expr, inferred)
            old = inferred.get(name)
            if new is None:
                merged = old
            elif old is None or old == new:
                merged = new
            elif {old, new} <= {"bool", "int64"}:
                merged = "int64"
            else:
                merged = "float64"
            if merged is not None and inferred.get(name) != merged:
                inferred[name] = merged
                changed = True
        if not changed:
            break
    return {name: inferred.get(name, "float64") for name in names}


__all__ = ["emit_cython_expr", "expression_dtype", "infer_local_types"]
