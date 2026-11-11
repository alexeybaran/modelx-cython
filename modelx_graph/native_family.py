from __future__ import annotations

import ast
import dis
import hashlib
import inspect
import math
import numbers
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .build import build_extension
from .loop_recovery import AffineIntBinding, ConstantBinding
from .native_analysis import NativePlanAnalysis
from .realized_compiler import RealizedTraceCompiler


class NativeFamilyError(RuntimeError):
    """Fail-closed error for the first real loop-family native slice."""


@dataclass(frozen=True)
class TypedFormulaSlots:
    formula_op_id: int
    dtype: str
    pool: str
    base: int
    slot_count: int


@dataclass(frozen=True)
class TypedNodeAddress:
    node_id: int
    formula_op_id: int
    pool: str
    offset: int


@dataclass(frozen=True)
class TypedSlotPlan:
    formulae: tuple[TypedFormulaSlots, ...]
    node_addresses: tuple[TypedNodeAddress, ...]
    double_count: int
    int_count: int
    bool_count: int
    object_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "_formula_by_id", {x.formula_op_id: x for x in self.formulae})
        object.__setattr__(self, "_node_by_id", {x.node_id: x for x in self.node_addresses})

    @property
    def formula_by_id(self) -> dict[int, TypedFormulaSlots]:
        return self._formula_by_id

    @property
    def node_by_id(self) -> dict[int, TypedNodeAddress]:
        return self._node_by_id


@dataclass(frozen=True)
class NativeFamilyProof:
    family_id: int
    formula_op_ids: tuple[int, ...]
    operation_count: int
    pyx_path: str
    so_path: str
    max_abs_error: float
    exact: bool
    native_median_s: float | None = None


@dataclass(frozen=True)
class _AddressExpr:
    pool: str
    kind: str
    a: int
    b: int = 0
    mod: int = 0

    def emit(self, arg: str | None = None) -> str:
        arr = {"double": "dslots", "int64": "islots", "bool": "bslots", "object": "oslots"}[self.pool]
        if self.kind == "constant":
            idx = str(self.a)
        elif self.kind == "affine":
            if arg is None:
                raise NativeFamilyError("affine address requires one argument")
            idx = f"({self.a} + ({self.b}) * ({arg}))"
        elif self.kind == "modulo":
            if arg is None:
                raise NativeFamilyError("modulo address requires one argument")
            idx = f"({self.a} + ((({arg}) - ({self.b})) % {self.mod}))"
        else:
            raise NativeFamilyError(f"unknown address expression {self.kind!r}")
        expr = f"{arr}[{idx}]"
        if self.pool == "bool":
            return f"({expr} != 0)"
        return expr


@dataclass(frozen=True)
class _ConcreteLiteralAddressExpr:
    """Exact finite key-to-slot map for one realized LiteralBlock formula."""

    pool: str
    rows: tuple[tuple[tuple[Any, ...], int], ...]

    def emit_args(self, args: tuple[str, ...]) -> str:
        arr = {"double": "dslots", "int64": "islots", "bool": "bslots", "object": "oslots"}[self.pool]
        branches: list[tuple[str, int]] = []
        for key, offset in self.rows:
            if len(key) != len(args):
                raise NativeFamilyError("literal reference key arity changed")
            tests = []
            for arg, raw_value in zip(args, key):
                value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
                if not isinstance(value, (bool, int, float, str)) and value is not None:
                    raise NativeFamilyError("literal reference key is not a scalar Cython literal")
                tests.append(f"(({arg}) == ({repr(value)}))")
            branches.append((" and ".join(tests) if tests else "1", int(offset)))
        index = "_missing_literal_reference()"
        for test, offset in reversed(branches):
            index = f"(({offset}) if ({test}) else ({index}))"
        expr = f"{arr}[{index}]"
        return f"({expr} != 0)" if self.pool == "bool" else expr


@dataclass(frozen=True)
class _CallResolver:
    name: str
    impl: Any
    address: Any
    arg_count: int


@dataclass(frozen=True)
class _ProjectedCallNode:
    args: tuple[Any, ...]


_BINOPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "**", ast.BitAnd: "&", ast.BitOr: "|",
    ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>",
}
_CMPOPS = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
}


def _parse_func(source: str) -> ast.FunctionDef:
    mod = ast.parse(textwrap.dedent(source))
    funcs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        raise NativeFamilyError("formula source has no function definition")
    return funcs[0]


def _impl_from_global(value: Any) -> Any | None:
    if inspect.ismethod(value):
        owner = getattr(value, "__self__", None)
        func = getattr(value, "__func__", None)
        if owner is not None and getattr(func, "__name__", None) == "call" and hasattr(owner, "data"):
            return owner
    if hasattr(value, "_impl") and value.__class__.__name__ == "Cells":
        return value._impl
    return None


def _typed_slot_plan(
    compiler: RealizedTraceCompiler,
    analysis: NativePlanAnalysis,
    *,
    include_objects: bool = False,
) -> TypedSlotPlan:
    if compiler.slots is None:
        raise NativeFamilyError("physical slot layout is required")
    by_analysis = {x.formula_op_id: x for x in analysis.formulae}
    formulae: list[TypedFormulaSlots] = []
    dbase = ibase = bbase = obase = 0
    for fl in sorted(compiler.slots.layout.formula_layouts, key=lambda x: x.formula_op_id):
        dtype = by_analysis[fl.formula_op_id].observed_return_type.dtype
        if dtype == "float64":
            pool, base = "double", dbase; dbase += fl.slot_count
        elif dtype == "int64":
            pool, base = "int64", ibase; ibase += fl.slot_count
        elif dtype == "bool":
            pool, base = "bool", bbase; bbase += fl.slot_count
        elif dtype == "object" and include_objects:
            pool, base = "object", obase; obase += fl.slot_count
        else:
            continue
        formulae.append(TypedFormulaSlots(fl.formula_op_id, dtype, pool, base, fl.slot_count))
    fby = {x.formula_op_id: x for x in formulae}
    nodes: list[TypedNodeAddress] = []
    for ns in compiler.slots.layout.node_slots:
        tf = fby.get(ns.formula_op_id)
        if tf is None:
            continue
        nodes.append(TypedNodeAddress(ns.node_id, ns.formula_op_id, tf.pool, tf.base + ns.slot_index))
    return TypedSlotPlan(tuple(formulae), tuple(nodes), dbase, ibase, bbase, obase)


def _fit_address(nodes: list[Any], addresses: list[TypedNodeAddress]) -> _AddressExpr:
    if not nodes or len(nodes) != len(addresses):
        raise NativeFamilyError("address resolver has no realized nodes")
    pools = {a.pool for a in addresses}
    if len(pools) != 1:
        raise NativeFamilyError("one Cell resolver spans multiple typed pools")
    pool = next(iter(pools))
    offsets = [a.offset for a in addresses]
    if len({len(n.args) for n in nodes}) != 1:
        raise NativeFamilyError("Cell arity changed within one realized resolver")
    arity = len(nodes[0].args)
    if arity == 0:
        if len(set(offsets)) != 1:
            raise NativeFamilyError("zero-argument Cell maps to multiple physical slots")
        return _AddressExpr(pool, "constant", offsets[0])
    if arity != 1 or not all(isinstance(n.args[0], (numbers.Integral, np.integer)) and not isinstance(n.args[0], (bool, np.bool_)) for n in nodes):
        raise NativeFamilyError("first native family slice supports zero-arg or one-integer-arg Cell reads")
    keys = [int(n.args[0]) for n in nodes]
    pairs = sorted(zip(keys, offsets))
    # Deduplicate exact key/address observations.
    mapping: dict[int, int] = {}
    for key, off in pairs:
        prev = mapping.setdefault(key, off)
        if prev != off:
            raise NativeFamilyError("same realized Cell key maps to several physical slots")
    keys = sorted(mapping)
    offs = [mapping[k] for k in keys]
    if len(set(offs)) == 1:
        return _AddressExpr(pool, "constant", offs[0])
    # Affine physical address in the actual argument value.
    if len(keys) >= 2:
        dk = keys[1] - keys[0]
        do = offs[1] - offs[0]
        if dk != 0 and do % dk == 0:
            slope = do // dk
            intercept = offs[0] - slope * keys[0]
            if all(intercept + slope * k == o for k, o in zip(keys, offs)):
                return _AddressExpr(pool, "affine", intercept, slope)
    # Common cyclic ring over a dense integer key domain.
    lo, hi = min(keys), max(keys)
    if keys == list(range(lo, hi + 1)):
        cap = len(set(offs))
        base = min(offs)
        if cap > 0 and all(o == base + ((k - lo) % cap) for k, o in zip(keys, offs)):
            return _AddressExpr(pool, "modulo", base, lo, cap)
    raise NativeFamilyError("Cell physical address is not constant/affine/simple-ring in the first native slice")


def _fit_literal_address(nodes: list[Any], addresses: list[TypedNodeAddress]) -> Any:
    """Fit a compact address or an exact finite multi-key literal map."""
    try:
        return _fit_address(nodes, addresses)
    except NativeFamilyError:
        pass
    if not nodes or len(nodes) != len(addresses):
        raise NativeFamilyError("literal address resolver has no realized nodes")
    pools = {address.pool for address in addresses}
    if len(pools) != 1:
        raise NativeFamilyError("literal Cell resolver spans multiple typed pools")
    arities = {len(node.args) for node in nodes}
    if len(arities) != 1:
        raise NativeFamilyError("literal Cell arity changed")
    mapping: dict[tuple[Any, ...], int] = {}
    for node, address in zip(nodes, addresses):
        key = tuple(value.item() if isinstance(value, np.generic) else value for value in node.args)
        if any(
            not isinstance(value, (bool, int, float, str)) and value is not None
            for value in key
        ):
            raise NativeFamilyError("literal Cell key contains a non-scalar value")
        previous = mapping.setdefault(key, int(address.offset))
        if previous != int(address.offset):
            raise NativeFamilyError("literal Cell key maps to several physical slots")
    if len(mapping) > 2048:
        raise NativeFamilyError("literal Cell finite address map exceeds 2048 entries")
    return _ConcreteLiteralAddressExpr(
        next(iter(pools)), tuple(sorted(mapping.items(), key=lambda item: repr(item[0])))
    )


def _output_address_expr(addresses: list[TypedNodeAddress]) -> _AddressExpr:
    if not addresses:
        raise NativeFamilyError("formula role has no output addresses")
    pool = addresses[0].pool
    if any(a.pool != pool for a in addresses):
        raise NativeFamilyError("formula output changes typed pool")
    offs = [a.offset for a in addresses]
    if len(set(offs)) == 1:
        return _AddressExpr(pool, "constant", offs[0])
    if len(offs) >= 2:
        step = offs[1] - offs[0]
        if all(offs[i] == offs[0] + step * i for i in range(len(offs))):
            return _AddressExpr(pool, "affine", offs[0], step)
    cap = len(set(offs)); base = min(offs)
    if cap and all(o == base + (i % cap) for i, o in enumerate(offs)):
        return _AddressExpr(pool, "modulo", base, 0, cap)
    raise NativeFamilyError("formula output slot stream is not constant/affine/simple-ring")


def _binding_expr(binding: Any, counter: str) -> tuple[str, str]:
    if isinstance(binding, ConstantBinding):
        v = binding.value
        if isinstance(v, (bool, np.bool_)):
            return ("bint", "1" if bool(v) else "0")
        if isinstance(v, (numbers.Integral, np.integer)):
            return ("long long", repr(int(v)))
        if isinstance(v, (numbers.Real, np.floating)):
            return ("double", repr(float(v)))
        raise NativeFamilyError("native formula binding contains a non-numeric constant")
    if isinstance(binding, AffineIntBinding):
        return ("long long", f"({int(binding.base)} + ({int(binding.step)}) * {counter})")
    raise NativeFamilyError(f"binding kind {getattr(binding, 'kind', type(binding).__name__)} not in first native slice")


class _FormulaEmitter:
    def __init__(self, compiler: RealizedTraceCompiler, typed: TypedSlotPlan, formula_op_id: int):
        self.compiler = compiler
        self.typed = typed
        self.formula_op_id = formula_op_id
        plan = compiler.structured.canonical_plan
        assert plan is not None
        self.formula_op = plan.formula_ops[formula_op_id]
        # Representative concrete node retains the exact altered-function globals.
        reps = [n for n in compiler.trace.nodes if n.shape_token == self.formula_op.role]
        if not reps:
            raise NativeFamilyError(f"no representative node for FormulaOp {formula_op_id}")
        self.rep = reps[0]
        self.fn = _parse_func(self.rep.schema.source)
        self.globals = self.rep.obj.altfunc.__globals__
        self.formals = {a.arg for a in self.fn.args.args}
        self.locals: set[str] = set()
        for n in ast.walk(self.fn):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if isinstance(t, ast.Name): self.locals.add(t.id)
        self.resolvers: dict[str, _CallResolver] = {}
        self._build_resolvers()

    def _build_resolvers(self) -> None:
        obj_to_nodes: dict[int, list[Any]] = defaultdict(list)
        for n in self.compiler.trace.nodes:
            if n.node_id in self.typed.node_by_id:
                obj_to_nodes[id(n.obj)].append(n)
        for name in {n.id for n in ast.walk(self.fn) if isinstance(n, ast.Name)}:
            if name not in self.globals:
                continue
            impl = _impl_from_global(self.globals[name])
            if impl is None:
                continue
            nodes = obj_to_nodes.get(id(impl), [])
            if not nodes:
                raise NativeFamilyError(f"Cell call {name} has no typed physical-slot nodes")
            addrs = [self.typed.node_by_id[n.node_id] for n in nodes]
            self.resolvers[name] = _CallResolver(name, impl, _fit_address(nodes, addrs), len(nodes[0].args))

    def _resolved_cell_call_expr(self, node: ast.Call, name: str) -> str | None:
        callsite_key = (
            name, int(getattr(node, "lineno", -1)), int(getattr(node, "col_offset", -1))
        )
        resolver = getattr(self, "callsite_resolvers", {}).get(callsite_key)
        if resolver is None:
            resolver = self.resolvers.get(name)
        if resolver is None:
            return None
        if node.keywords or len(node.args) != resolver.arg_count:
            raise NativeFamilyError(f"Cell call {name} argument shape changed")
        emitted_args = tuple(self.expr(arg) for arg in node.args)
        if hasattr(resolver.address, "emit_args"):
            return resolver.address.emit_args(emitted_args)
        if getattr(resolver.address, "occurrence_addressed", False):
            return resolver.address.emit()
        if resolver.arg_count == 0:
            return resolver.address.emit()
        if resolver.arg_count == 1:
            return resolver.address.emit(emitted_args[0])
        raise NativeFamilyError("multi-parameter Cell call unsupported")

    def _expression_is_native_numeric(self, node: ast.AST) -> bool:
        """Return True when a numeric conversion source is already C-numeric."""
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (bool, int, float))
        if isinstance(node, ast.Name):
            typ = getattr(self, "formal_cython_types", {}).get(node.id)
            if typ in {"double", "long long", "bint"}:
                return True
            if node.id in self.globals:
                value = self.globals[node.id]
                return isinstance(
                    value,
                    (bool, np.bool_, numbers.Integral, np.integer, numbers.Real, np.floating),
                )
            return False
        if isinstance(node, ast.UnaryOp):
            return self._expression_is_native_numeric(node.operand)
        if isinstance(node, ast.BinOp):
            return self._expression_is_native_numeric(node.left) and self._expression_is_native_numeric(node.right)
        if isinstance(node, ast.IfExp):
            return self._expression_is_native_numeric(node.body) and self._expression_is_native_numeric(node.orelse)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            resolver = self.resolvers.get(name)
            if resolver is not None:
                return getattr(resolver.address, "pool", None) in {"double", "int64", "bool"}
            if name in {"abs", "min", "max", "round"}:
                return all(self._expression_is_native_numeric(arg) for arg in node.args)
            if name in {"int", "float", "bool"} and len(node.args) == 1:
                # The result is numeric even when the source needs semantic text
                # conversion via _mxg_int/_mxg_float.
                return True
        return False

    def expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool): return "1" if node.value else "0"
            if isinstance(node.value, (int, float, str)): return repr(node.value)
            if node.value is None: return "None"
            raise NativeFamilyError(f"unsupported constant {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in self.formals or node.id in self.locals: return node.id
            if node.id in ("True", "False"): return "1" if node.id == "True" else "0"
            if node.id in self.globals:
                v = self.globals[node.id]
                if isinstance(v, (bool, np.bool_)): return "1" if bool(v) else "0"
                if isinstance(v, (numbers.Integral, np.integer)): return repr(int(v))
                if isinstance(v, (numbers.Real, np.floating)): return repr(float(v))
                if isinstance(v, str): return repr(v)
                if v is None: return "None"
            object_globals = getattr(self, "object_globals", {})
            if node.id in object_globals:
                return f"_mxg_engine_globals[{int(object_globals[node.id])}]"
            raise NativeFamilyError(f"unsupported name {node.id}")
        if isinstance(node, ast.Tuple):
            inner = ", ".join(self.expr(x) for x in node.elts)
            if len(node.elts) == 1: inner += ","
            return f"({inner})"
        if isinstance(node, ast.List):
            return "[" + ", ".join(self.expr(x) for x in node.elts) + "]"
        if isinstance(node, ast.Dict):
            return "{" + ", ".join(
                f"{self.expr(key)}: {self.expr(value)}"
                for key, value in zip(node.keys, node.values)
                if key is not None
            ) + "}"
        if isinstance(node, ast.Attribute):
            return f"({self.expr(node.value)}).{node.attr}"
        if isinstance(node, ast.Subscript):
            # Generated modules use ``wraparound=False`` for numeric-array speed.
            # Cython also applies that directive to inferred Python lists, where
            # a direct ``items[-1]`` then becomes unchecked undefined behavior.
            # Force Python object indexing for a literal negative subscript so
            # modelx-cython formula semantics remain exact without weakening the
            # bounded-array hot path globally.
            if (
                isinstance(node.slice, ast.UnaryOp)
                and isinstance(node.slice.op, ast.USub)
                and isinstance(node.slice.operand, ast.Constant)
                and isinstance(node.slice.operand.value, int)
            ):
                return (
                    f"(<object>({self.expr(node.value)}))"
                    f"[(<object>({self.expr(node.slice)}))]"
                )
            # In a subscript, an AST tuple containing Slice nodes represents
            # multidimensional indexing (a[:, i]), not an ordinary tuple value.
            # Emitting the generic tuple form would produce invalid Cython such
            # as a[(:, i)].  Keep the indexing syntax exactly at the object/NumPy
            # boundary instead of trying to scalarize it.
            if isinstance(node.slice, ast.Tuple):
                index = ", ".join(self.expr(item) for item in node.slice.elts)
            else:
                index = self.expr(node.slice)
            return f"({self.expr(node.value)})[{index}]"
        if isinstance(node, ast.Slice):
            lower = "" if node.lower is None else self.expr(node.lower)
            upper = "" if node.upper is None else self.expr(node.upper)
            step = "" if node.step is None else self.expr(node.step)
            return f"{lower}:{upper}" + (f":{step}" if node.step is not None else "")
        if isinstance(node, ast.ListComp):
            parts = []
            for gen in node.generators:
                if gen.is_async or not isinstance(gen.target, ast.Name):
                    raise NativeFamilyError("unsupported comprehension target")
                text = f"for {gen.target.id} in {self.expr(gen.iter)}"
                for cond in gen.ifs:
                    text += f" if {self.expr(cond)}"
                parts.append(text)
            return "[" + self.expr(node.elt) + " " + " ".join(parts) + "]"
        if isinstance(node, ast.GeneratorExp):
            parts = []
            for gen in node.generators:
                if gen.is_async or not isinstance(gen.target, ast.Name):
                    raise NativeFamilyError("unsupported generator target")
                text = f"for {gen.target.id} in {self.expr(gen.iter)}"
                for cond in gen.ifs:
                    text += f" if {self.expr(cond)}"
                parts.append(text)
            return "(" + self.expr(node.elt) + " " + " ".join(parts) + ")"
        if isinstance(node, ast.UnaryOp):
            inner = self.expr(node.operand)
            if isinstance(node.op, ast.USub): return f"(-({inner}))"
            if isinstance(node.op, ast.UAdd): return f"(+({inner}))"
            if isinstance(node.op, ast.Not): return f"(not ({inner}))"
            raise NativeFamilyError(f"unsupported unary {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            left, right = self.expr(node.left), self.expr(node.right)
            if isinstance(node.op, ast.Div): return f"((<double>({left})) / (<double>({right})))"
            if isinstance(node.op, ast.Pow):
                # Cython otherwise treats an integral base with a fractional
                # exponent as potentially complex and rejects comparisons with
                # the result.  modelx-cython's numeric formula ABI is a C double.
                return f"c_pow((<double>({left})), (<double>({right})))"
            op = _BINOPS.get(type(node.op))
            if op is None: raise NativeFamilyError(f"unsupported binary {type(node.op).__name__}")
            return f"(({left}) {op} ({right}))"
        if isinstance(node, ast.BoolOp):
            op = " and " if isinstance(node.op, ast.And) else " or " if isinstance(node.op, ast.Or) else None
            if op is None: raise NativeFamilyError("unsupported boolean operator")
            return "(" + op.join(f"({self.expr(v)})" for v in node.values) + ")"
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            operators: list[str] = []
            for raw_op in node.ops:
                if isinstance(raw_op, ast.In):
                    op = "in"
                elif isinstance(raw_op, ast.NotIn):
                    op = "not in"
                elif isinstance(raw_op, ast.Is):
                    op = "is"
                elif isinstance(raw_op, ast.IsNot):
                    op = "is not"
                else:
                    op = _CMPOPS.get(type(raw_op))
                if op is None:
                    raise NativeFamilyError(f"unsupported comparison {type(raw_op).__name__}")
                operators.append(op)
            parts = [f"({self.expr(operands[0])})"]
            for op, operand in zip(operators, operands[1:]):
                parts.extend([op, f"({self.expr(operand)})"])
            return "(" + " ".join(parts) + ")"
        if isinstance(node, ast.IfExp):
            return f"(({self.expr(node.body)}) if ({self.expr(node.test)}) else ({self.expr(node.orelse)}))"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            args = [self.expr(x) for x in node.args]
            args.extend(f"{kw.arg}={self.expr(kw.value)}" for kw in node.keywords if kw.arg is not None)
            if any(kw.arg is None for kw in node.keywords):
                raise NativeFamilyError("**kwargs are unsupported in engine formulas")
            return f"{self.expr(node.func)}(" + ", ".join(args) + ")"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            resolved = self._resolved_cell_call_expr(node, name)
            if resolved is not None:
                return resolved
            if (
                name in getattr(self, "object_globals", {})
                and not node.args
                and not node.keywords
                and name in self.globals
                and type(self.globals[name]).__module__.startswith("modelx.io.")
            ):
                # modelx IO references are snapshotted into detached plain
                # Python objects. Their zero-argument interface call means
                # "return the payload", so emit the payload directly.
                return self.expr(node.func)
            if name in {"float", "int", "bool"} and len(node.args) == 1 and not node.keywords:
                inner = self.expr(node.args[0])
                if name == "bool":
                    return f"(<bint>({inner}))"
                if self._expression_is_native_numeric(node.args[0]):
                    cast = "double" if name == "float" else "long long"
                    return f"(<{cast}>({inner}))"
                # Cython's raw object casts already implement Python numeric
                # protocols for normal numeric/NumPy/Decimal objects, but reject
                # text/bytes even though Python int()/float() accept them. The
                # helper keeps the fast raw cast and adds only that semantic gap.
                helper = "_mxg_float" if name == "float" else "_mxg_int"
                return f"{helper}({inner})"
            if name == "abs" and len(node.args) == 1 and not node.keywords:
                inner = self.expr(node.args[0]); return f"(({inner}) if ({inner}) >= 0 else -({inner}))"
            if name in {"min", "max"} and len(node.args) == 2 and not node.keywords:
                a,b=self.expr(node.args[0]),self.expr(node.args[1]); cmp="<" if name=="min" else ">"
                return f"(({a}) if ({a}) {cmp} ({b}) else ({b}))"
            if name in {"min", "max"} and len(node.args) > 2 and not node.keywords:
                values = [self.expr(arg) for arg in node.args]
                cmp = "<" if name == "min" else ">"
                result = values[0]
                for value in values[1:]:
                    result = f"(({result}) if ({result}) {cmp} ({value}) else ({value}))"
                return result
            if name in {"min", "max"} and len(node.args) == 1 and not node.keywords:
                return f"{name}({self.expr(node.args[0])})"
            if name == "round" and len(node.args) in (1, 2) and not node.keywords:
                return "round(" + ", ".join(self.expr(x) for x in node.args) + ")"
            if name == "str" and len(node.args) == 1 and not node.keywords:
                return "str(" + self.expr(node.args[0]) + ")"
            if name in {"range", "len", "sum", "list", "tuple"} and not node.keywords:
                return name + "(" + ", ".join(self.expr(x) for x in node.args) + ")"
            if name in {"sorted", "hasattr"}:
                args = []
                for arg in node.args:
                    if name == "sorted" and isinstance(arg, ast.GeneratorExp):
                        args.append(self.expr(ast.ListComp(elt=arg.elt, generators=arg.generators)))
                    else:
                        args.append(self.expr(arg))
                args.extend(f"{kw.arg}={self.expr(kw.value)}" for kw in node.keywords if kw.arg is not None)
                if any(kw.arg is None for kw in node.keywords):
                    raise NativeFamilyError(f"**kwargs are unsupported for {name}")
                return name + "(" + ", ".join(args) + ")"
            if name in getattr(self, "object_globals", {}):
                args = [self.expr(x) for x in node.args]
                args.extend(f"{kw.arg}={self.expr(kw.value)}" for kw in node.keywords if kw.arg is not None)
                if any(kw.arg is None for kw in node.keywords):
                    raise NativeFamilyError("**kwargs are unsupported in engine formulas")
                return f"{self.expr(node.func)}(" + ", ".join(args) + ")"
            raise NativeFamilyError(f"unsupported call {name}")
        raise NativeFamilyError(f"unsupported expression {ast.dump(node, include_attributes=False)}")

    def target(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, (ast.Tuple, ast.List)):
            inner = ", ".join(self.target(item) for item in node.elts)
            return f"({inner})"
        raise NativeFamilyError(f"unsupported assignment target {type(node).__name__}")

    def statements(self, body: list[ast.stmt], indent: str = "    ") -> list[str]:
        out: list[str] = []
        for st in body:
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                continue
            if isinstance(st, ast.Return):
                out.append(indent + "return " + self.expr(st.value))
            elif isinstance(st, ast.Assign) and len(st.targets)==1 and isinstance(st.targets[0],ast.Name):
                out.append(indent + f"{st.targets[0].id} = {self.expr(st.value)}")
            elif isinstance(st, ast.Assign) and len(st.targets) == 1:
                out.append(indent + f"{self.target(st.targets[0])} = {self.expr(st.value)}")
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
                out.append(indent + f"{st.target.id} = {self.expr(st.value)}")
            elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                op=_BINOPS.get(type(st.op))
                if op is None: raise NativeFamilyError("unsupported augmented operator")
                out.append(indent + f"{st.target.id} {op}= {self.expr(st.value)}")
            elif isinstance(st, ast.If):
                out.append(indent + "if " + self.expr(st.test) + ":")
                out.extend(self.statements(st.body, indent+"    "))
                if st.orelse:
                    out.append(indent + "else:")
                    out.extend(self.statements(st.orelse, indent+"    "))
            elif isinstance(st, ast.For):
                if st.orelse:
                    raise NativeFamilyError("for-else is unsupported in engine formulas")
                out.append(indent + f"for {self.target(st.target)} in {self.expr(st.iter)}:")
                out.extend(self.statements(st.body, indent + "    "))
            elif isinstance(st, ast.Break):
                out.append(indent + "break")
            elif isinstance(st, ast.Continue):
                out.append(indent + "continue")
            elif isinstance(st, ast.Raise):
                if st.exc is None:
                    out.append(indent + "raise")
                elif isinstance(st.exc, ast.Call) and isinstance(st.exc.func, ast.Name) and st.exc.func.id in {"ValueError", "TypeError", "RuntimeError"}:
                    out.append(indent + "raise " + st.exc.func.id + "(" + ", ".join(self.expr(x) for x in st.exc.args) + ")")
                else:
                    raise NativeFamilyError("unsupported native raise")
            elif isinstance(st, ast.Pass):
                out.append(indent + "pass")
            elif isinstance(st, ast.Expr):
                out.append(indent + self.expr(st.value))
            else:
                raise NativeFamilyError(f"unsupported native statement {type(st).__name__}")
        return out

    def emit_function(self, arg_types: list[tuple[str,str]], return_dtype: str) -> list[str]:
        rettype={"float64":"double","int64":"long long","bool":"bint"}[return_dtype]
        args=", ".join(f"{typ} {name}" for name,typ in arg_types)
        if args: args += ", "
        lines=[f"cdef inline {rettype} f_{self.formula_op_id}({args}double[::1] dslots, long long[::1] islots, unsigned char[::1] bslots):"]
        for name in sorted(self.locals - self.formals):
            lines.append(f"    cdef double {name} = 0.0")
        lines.extend(self.statements(self.fn.body, "    "))
        return lines


def _role_node_sequences(compiler: RealizedTraceCompiler, family_id: int):
    plan=compiler.structured.canonical_plan
    assert plan is not None
    family=plan.code_families[family_id]
    instances=[x for x in plan.loop_instances if x.family_id==family_id]
    if len(instances)!=1:
        raise NativeFamilyError("first native family slice requires one loop instance per family")
    inst=instances[0]
    expanded=family.expanded_variants()
    if len(expanded)!=1 or any(pid!=0 for pid in inst.variant_ids):
        raise NativeFamilyError("first native family slice requires one observed variant")
    role_count=len(family.formula_op_ids)
    counters=[0]*role_count
    seqs=[[] for _ in range(role_count)]
    op_sequence=[]
    rt_to_id=compiler.trace.runtime_to_id
    pattern=expanded[0]
    for _pid in inst.variant_ids:
        for role in pattern:
            rb=inst.role_bindings[role]
            if rb is None: raise NativeFamilyError("family terminal has no binding")
            occ=counters[role]
            runtime=rb.runtime_node(occ)
            try: nid=rt_to_id[runtime]
            except KeyError as exc: raise NativeFamilyError("family binding runtime node missing from trace") from exc
            seqs[role].append(nid); op_sequence.append((role,nid)); counters[role]+=1
    return family, inst, tuple(tuple(x) for x in seqs), tuple(op_sequence)


def _authoritative_value(node: Any) -> Any:
    try:
        return node.obj.data[node.args]
    except Exception as exc:
        raise NativeFamilyError(f"authoritative cached value unavailable for {node.uid}") from exc


def build_native_family_cython(
    compiler: RealizedTraceCompiler,
    family_id: int,
    build_dir: str | Path,
    *,
    module_prefix: str = "mxg_native_family",
) -> NativeFamilyProof:
    """Compile one fully native recovered loop family against real v0.15 slots.

    This first slice deliberately supports one loop instance / one variant and
    numeric constant/affine bindings. Unsupported families remain Python. The
    formula call resolver is derived from concrete CellsImpl identity and the
    PhysicalSlotLayout, never from Cell names or a privileged coordinate.
    """
    if compiler.slots is None:
        compiler.lower_slots()
    analysis=compiler.native_plan or compiler.analyze_native()
    typed=_typed_slot_plan(compiler,analysis)
    family,inst,role_nodes,op_sequence=_role_node_sequences(compiler,family_id)
    aby={x.formula_op_id:x for x in analysis.formulae}
    if any(not aby[fid].native_capable for fid in set(family.formula_op_ids)):
        bad=[(fid,aby[fid].name,aby[fid].reasons) for fid in set(family.formula_op_ids) if not aby[fid].native_capable]
        raise NativeFamilyError(f"family contains Python FormulaOps: {bad}")
    if any(fid not in typed.formula_by_id for fid in set(family.formula_op_ids)):
        raise NativeFamilyError("family contains non-numeric FormulaOp")

    # Role metadata and output slot streams.
    role_meta=[]
    emitters={}
    plan=compiler.structured.canonical_plan; assert plan is not None
    for role,fid in enumerate(family.formula_op_ids):
        rb=inst.role_bindings[role]
        if rb is None: raise NativeFamilyError("missing family role binding")
        op=plan.formula_ops[fid]
        args=[]
        for formal,binding in zip(_parse_func(next(n for n in compiler.trace.nodes if n.shape_token==op.role).schema.source).args.args, rb.arg_bindings):
            typ,expr=_binding_expr(binding,f"c_{role}")
            args.append((formal.arg,typ,expr))
        nodes=[compiler.sequential.node_by_id[nid] for nid in role_nodes[role]]
        addrs=[typed.node_by_id[n.node_id] for n in nodes]
        out_addr=_output_address_expr(addrs)
        emitter=_FormulaEmitter(compiler,typed,fid); emitters[fid]=emitter
        role_meta.append((fid,args,out_addr,nodes))

    # Exact external numeric dependencies needed to seed the shared pools.
    family_node_ids={nid for _r,nid in op_sequence}
    deps_by_dst=defaultdict(list)
    for src,dst in compiler.trace.dependencies:
        deps_by_dst[dst].append(src)
    external=set()
    for _role,nid in op_sequence:
        for src in deps_by_dst.get(nid,()):
            if src not in family_node_ids:
                external.add(src)
    dslots=np.zeros(typed.double_count,dtype=np.float64)
    islots=np.zeros(typed.int_count,dtype=np.int64)
    bslots=np.zeros(typed.bool_count,dtype=np.uint8)
    occupied={"double":{},"int64":{},"bool":{}}
    byid=compiler.sequential.node_by_id
    for nid in sorted(external):
        addr=typed.node_by_id.get(nid)
        if addr is None:
            raise NativeFamilyError(f"external dependency node {nid} has no typed physical address")
        value=_authoritative_value(byid[nid])
        prior=occupied[addr.pool].get(addr.offset)
        scalar=float(value) if addr.pool=="double" else int(value)
        if prior is not None and prior != scalar:
            raise NativeFamilyError("external dependencies require colliding bounded slots; family needs integrated execution")
        occupied[addr.pool][addr.offset]=scalar
        if addr.pool=="double": dslots[addr.offset]=scalar
        elif addr.pool=="int64": islots[addr.offset]=scalar
        else: bslots[addr.offset]=1 if bool(value) else 0

    # Generate compact serial Cython. Role counters are separate from loop index so
    # the shape extends naturally to future multi-variant control streams.
    build_dir=Path(build_dir); build_dir.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha1((str(family_id)+"|"+"|".join(plan.formula_ops[f].schema_uid for f in family.formula_op_ids)).encode()).hexdigest()[:12]
    module_name=f"{module_prefix}_{digest}"
    pyx=build_dir/f"{module_name}.pyx"
    lines=[
        "# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False",
        "import numpy as np",
        "cimport numpy as cnp",
        "from libc.math cimport pow as c_pow",
        "",
    ]
    lines += _emit_python_numeric_conversion_helpers()
    for fid in dict.fromkeys(family.formula_op_ids):
        meta=aby[fid]
        # signature types come from the family role using this FormulaOp.
        role=family.formula_op_ids.index(fid)
        args=[(name,typ) for name,typ,_expr in role_meta[role][1]]
        lines.extend(emitters[fid].emit_function(args,meta.observed_return_type.dtype)); lines.append("")
    lines += [
        "cpdef cnp.ndarray run(cnp.ndarray[cnp.float64_t, ndim=1] d_arr, cnp.ndarray[cnp.int64_t, ndim=1] i_arr, cnp.ndarray[cnp.uint8_t, ndim=1] b_arr):",
        "    cdef double[::1] dslots = d_arr",
        "    cdef long long[::1] islots = i_arr",
        "    cdef unsigned char[::1] bslots = b_arr",
        f"    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_arr = np.empty({len(op_sequence)}, dtype=np.float64)",
        "    cdef double[::1] out = out_arr",
        "    cdef Py_ssize_t iteration",
        "    cdef Py_ssize_t out_pos = 0",
    ]
    for role in range(len(family.formula_op_ids)):
        lines.append(f"    cdef Py_ssize_t c_{role} = 0")
    lines.append(f"    for iteration in range({len(inst.variant_ids)}):")
    pattern=family.expanded_variants()[0]
    for role in pattern:
        fid,args,out_addr,_nodes=role_meta[role]
        arg_exprs=[expr for _name,_typ,expr in args]
        callargs=", ".join(arg_exprs + ["dslots","islots","bslots"])
        value=f"f_{fid}({callargs})"
        # Output address uses occurrence counter as its argument for affine/modulo forms.
        if out_addr.kind=="constant":
            target=out_addr.emit()
        else:
            target=out_addr.emit(f"c_{role}")
        # bool output address.emit() returns comparison, unsuitable LHS; construct raw LHS.
        arr={"double":"dslots","int64":"islots","bool":"bslots"}[out_addr.pool]
        if out_addr.kind=="constant": idx=str(out_addr.a)
        elif out_addr.kind=="affine": idx=f"({out_addr.a} + ({out_addr.b}) * c_{role})"
        else: idx=f"({out_addr.a} + ((c_{role} - ({out_addr.b})) % {out_addr.mod}))"
        lines.append(f"        {arr}[{idx}] = {value}")
        cast=f"<double>{arr}[{idx}]"
        lines.append(f"        out[out_pos] = {cast}")
        lines.append("        out_pos += 1")
        lines.append(f"        c_{role} += 1")
    lines.append("    return out_arr")
    pyx.write_text("\n".join(lines)+"\n")
    mod,so_path,_proc=build_extension(pyx,module_name,build_dir,openmp=False,optimization="O2",native_arch=False)
    actual=np.asarray(mod.run(dslots.copy(),islots.copy(),bslots.copy()),dtype=float)
    expected=np.asarray([float(_authoritative_value(byid[nid])) for _role,nid in op_sequence],dtype=float)
    if actual.shape!=expected.shape: raise NativeFamilyError("native family output length mismatch")
    diff=np.abs(actual-expected)
    max_abs=float(diff.max()) if diff.size else 0.0
    exact=bool(np.array_equal(actual,expected))
    return NativeFamilyProof(family_id,tuple(family.formula_op_ids),len(op_sequence),str(pyx),str(so_path),max_abs,exact)

# ---------------------------------------------------------------------------
# First real mixed-native family slice
# ---------------------------------------------------------------------------

import types


class _PythonTypedCellResolver:
    __slots__ = ("mapping", "dslots", "islots", "bslots", "oslots", "original")

    def __init__(self, mapping, dslots, islots, bslots, original, oslots=None):
        self.mapping = mapping
        self.dslots = dslots
        self.islots = islots
        self.bslots = bslots
        self.oslots = oslots
        self.original = original

    def __call__(self, *args, **kwargs):
        if kwargs:
            return self.original(*args, **kwargs)
        try:
            addr = self.mapping[tuple(args)]
        except (KeyError, TypeError):
            return self.original(*args, **kwargs)
        if addr.pool == "double":
            return float(self.dslots[addr.offset])
        if addr.pool == "int64":
            return int(self.islots[addr.offset])
        if addr.pool == "bool":
            return bool(self.bslots[addr.offset])
        if addr.pool == "object" and self.oslots is not None:
            return self.oslots[addr.offset]
        return self.original(*args, **kwargs)


def _python_boundary_callback(
    compiler: RealizedTraceCompiler,
    typed: TypedSlotPlan,
    formula_op_id: int,
    dslots: np.ndarray,
    islots: np.ndarray,
    bslots: np.ndarray,
):
    plan=compiler.structured.canonical_plan; assert plan is not None
    op=plan.formula_ops[formula_op_id]
    reps=[n for n in compiler.trace.nodes if n.shape_token==op.role]
    if not reps: raise NativeFamilyError("Python boundary has no representative node")
    rep=reps[0]; fn=rep.obj.altfunc
    glb=dict(fn.__globals__)
    obj_nodes: dict[int,list[Any]]=defaultdict(list)
    for n in compiler.trace.nodes:
        if n.node_id in typed.node_by_id:
            obj_nodes[id(n.obj)].append(n)
    for name in set(fn.__code__.co_names):
        if name not in glb: continue
        original=glb[name]
        impl=_impl_from_global(original)
        if impl is None: continue
        nodes=obj_nodes.get(id(impl),[])
        if not nodes:
            # Leave genuinely non-typed/unscheduled helpers to Python/modelx.
            continue
        mapping={n.args:typed.node_by_id[n.node_id] for n in nodes}
        glb[name]=_PythonTypedCellResolver(mapping,dslots,islots,bslots,original)
    cloned=types.FunctionType(fn.__code__,glb,fn.__name__,fn.__defaults__,fn.__closure__)
    cloned.__kwdefaults__=getattr(fn,"__kwdefaults__",None)
    return cloned


def _python_boundary_callback_for_instance_role(
    compiler: RealizedTraceCompiler,
    typed: TypedSlotPlan,
    formula_op_id: int,
    node_ids: tuple[int, ...],
    dslots: np.ndarray,
    islots: np.ndarray,
    bslots: np.ndarray,
    oslots: list[Any] | None = None,
):
    """Build a Python callback from the concrete environment of one family instance.

    The same FormulaOp source may be bound to different concrete CellsImpl objects
    in different parameterized itemspaces. One callback is therefore prepared per
    role/instance rather than reused from an arbitrary representative environment.
    If a single role changes concrete formula object within one instance, this proof
    path fails closed instead of guessing a dispatch key.
    """
    if not node_ids:
        return None
    byid = compiler.sequential.node_by_id
    objs = {id(byid[nid].obj): byid[nid].obj for nid in node_ids}
    if len(objs) != 1:
        raise NativeFamilyError(
            f"Python FormulaOp {formula_op_id} changes concrete formula environment within one family instance"
        )
    obj = next(iter(objs.values()))
    return _python_boundary_callback_for_concrete_object(
        compiler, typed, obj, dslots, islots, bslots, oslots
    )


def _python_boundary_callback_for_concrete_object(
    compiler: RealizedTraceCompiler,
    typed: TypedSlotPlan,
    obj: Any,
    dslots: np.ndarray,
    islots: np.ndarray,
    bslots: np.ndarray,
    oslots: list[Any] | None = None,
):
    fn = obj.altfunc
    if any(ins.opname in {"STORE_GLOBAL", "DELETE_GLOBAL"} for ins in dis.get_instructions(fn)):
        raise NativeFamilyError(
            "Python boundary formula mutates globals; integrated DirectSlot fallback is required"
        )
    glb = dict(fn.__globals__)
    obj_nodes: dict[int, list[Any]] = defaultdict(list)
    for n in compiler.trace.nodes:
        if n.node_id in typed.node_by_id:
            obj_nodes[id(n.obj)].append(n)
    for name in set(fn.__code__.co_names):
        if name not in glb:
            continue
        original = glb[name]
        impl = _impl_from_global(original)
        if impl is None:
            continue
        nodes = obj_nodes.get(id(impl), [])
        if not nodes:
            continue
        mapping = {n.args: typed.node_by_id[n.node_id] for n in nodes}
        glb[name] = _PythonTypedCellResolver(
            mapping, dslots, islots, bslots, original, oslots
        )
    cloned = types.FunctionType(fn.__code__, glb, fn.__name__, fn.__defaults__, fn.__closure__)
    cloned.__kwdefaults__ = getattr(fn, "__kwdefaults__", None)
    cloned.__annotations__ = getattr(fn, "__annotations__", {}).copy()
    return cloned


def _python_boundary_callbacks_for_instance_role(
    compiler: RealizedTraceCompiler,
    typed: TypedSlotPlan,
    node_ids: tuple[int, ...],
    dslots: np.ndarray,
    islots: np.ndarray,
    bslots: np.ndarray,
    oslots: list[Any] | None = None,
) -> tuple[Any, ...]:
    """Return one callback reference per realized role occurrence.

    A recovered role may cross several parameterized/inherited concrete Cells
    objects inside one loop instance. Callback *code* is still shared, but Python
    globals must be bound to the concrete environment used by that occurrence.
    Keeping this selector stream as runtime data prevents model/itemspace names
    from leaking into generated Cython source. Duplicate environments reuse the
    same cloned function object.
    """
    byid = compiler.sequential.node_by_id
    cache: dict[int, Any] = {}
    out: list[Any] = []
    for nid in node_ids:
        obj = byid[nid].obj
        token = id(obj)
        callback = cache.get(token)
        if callback is None:
            callback = _python_boundary_callback_for_concrete_object(
                compiler, typed, obj, dslots, islots, bslots, oslots
            )
            cache[token] = callback
        out.append(callback)
    return tuple(out)


@dataclass(frozen=True)
class MixedNativeFamilyProof:
    family_id: int
    formula_op_ids: tuple[int, ...]
    native_formula_ops: int
    python_formula_ops: int
    operation_count: int
    python_callback_operations: int
    pyx_path: str
    so_path: str
    max_abs_error: float
    exact: bool


def _initial_typed_pools(compiler, typed, family_node_ids):
    deps_by_dst=defaultdict(list)
    for src,dst in compiler.trace.dependencies:
        deps_by_dst[dst].append(src)
    external=set()
    for nid in family_node_ids:
        for src in deps_by_dst.get(nid,()):
            if src not in family_node_ids: external.add(src)
    dslots=np.zeros(typed.double_count,dtype=np.float64)
    islots=np.zeros(typed.int_count,dtype=np.int64)
    bslots=np.zeros(typed.bool_count,dtype=np.uint8)
    occupied={"double":{},"int64":{},"bool":{}}
    byid=compiler.sequential.node_by_id
    for nid in sorted(external):
        addr=typed.node_by_id.get(nid)
        if addr is None:
            # Object-only external dependencies are legal only for Python
            # boundaries; they remain in ordinary Python globals/modelx state.
            continue
        if addr.pool == "object":
            # The Stage 4b.2 hybrid path seeds object state separately so the
            # established numeric-only v0.17 proof remains byte-for-byte stable.
            continue
        value=_authoritative_value(byid[nid])
        scalar=float(value) if addr.pool=="double" else int(value)
        prior=occupied[addr.pool].get(addr.offset)
        if prior is not None and prior != scalar:
            raise NativeFamilyError("external dependencies collide in a bounded slot; integrated predecessor execution is required")
        occupied[addr.pool][addr.offset]=scalar
        if addr.pool=="double": dslots[addr.offset]=scalar
        elif addr.pool=="int64": islots[addr.offset]=scalar
        else: bslots[addr.offset]=1 if bool(value) else 0
    return dslots,islots,bslots


def _initial_hybrid_pools(compiler, typed, family_node_ids):
    """Seed numeric and object slot pools for one isolated family proof.

    The whole-artifact executor will eventually obtain predecessor values from
    earlier regions rather than authoritative modelx caches.  Family-level proofs
    still need external dependency seeding, exactly as the v0.17 numeric proof
    does, so object dependencies use the same conservative collision guard here.
    """
    dslots, islots, bslots = _initial_typed_pools(compiler, typed, family_node_ids)
    oslots: list[Any] = [None] * typed.object_count
    if not typed.object_count:
        return dslots, islots, bslots, oslots

    deps_by_dst = defaultdict(list)
    for src, dst in compiler.trace.dependencies:
        deps_by_dst[dst].append(src)
    external = set()
    for nid in family_node_ids:
        for src in deps_by_dst.get(nid, ()):
            if src not in family_node_ids:
                external.add(src)

    occupied: dict[int, Any] = {}
    byid = compiler.sequential.node_by_id
    for nid in sorted(external):
        addr = typed.node_by_id.get(nid)
        if addr is None or addr.pool != "object":
            continue
        value = _authoritative_value(byid[nid])
        if addr.offset in occupied:
            prior = occupied[addr.offset]
            # Identity is the only universally safe equality relation for
            # arbitrary Python objects.  Different live objects sharing one
            # physical slot means isolated family seeding is insufficient and
            # predecessor execution must be integrated instead.
            if prior is not value:
                raise NativeFamilyError(
                    "external object dependencies collide in a bounded slot; "
                    "integrated predecessor execution is required"
                )
        occupied[addr.offset] = value
        oslots[addr.offset] = value
    return dslots, islots, bslots, oslots


def build_mixed_native_family_cython(
    compiler: RealizedTraceCompiler,
    family_id: int,
    build_dir: str | Path,
    *,
    module_prefix: str = "mxg_mixed_family",
) -> MixedNativeFamilyProof:
    """Compile one exact real family with numeric Python FormulaOp islands.

    Native FormulaOps read/write typed physical pools directly. Python FormulaOps
    execute their original Python bytecode; only their direct modelx Cell globals
    are rebound to readers over the *same* typed pools. This is the first real
    real-model proof of a Python island inside the recovered Cython loop.
    """
    if compiler.slots is None: compiler.lower_slots()
    analysis=compiler.native_plan or compiler.analyze_native()
    typed=_typed_slot_plan(compiler,analysis)
    family,inst,role_nodes,op_sequence=_role_node_sequences(compiler,family_id)
    aby={x.formula_op_id:x for x in analysis.formulae}
    plan=compiler.structured.canonical_plan; assert plan is not None
    # First mixed slice requires numeric outputs even for Python boundaries so the
    # shared ABI stays three typed pools. Arbitrary object pools remain a later
    # extension; the foundation already proved object callbacks mechanically.
    for fid in set(family.formula_op_ids):
        if fid not in typed.formula_by_id:
            raise NativeFamilyError(f"FormulaOp {fid} has object output; first real mixed slice keeps it Python-only")

    role_meta=[]; emitters={}
    for role,fid in enumerate(family.formula_op_ids):
        rb=inst.role_bindings[role]
        if rb is None: raise NativeFamilyError("missing family role binding")
        op=plan.formula_ops[fid]
        fn=_parse_func(next(n for n in compiler.trace.nodes if n.shape_token==op.role).schema.source)
        if len(fn.args.args)!=len(rb.arg_bindings):
            raise NativeFamilyError("formal/binding arity mismatch")
        args=[]
        for formal,binding in zip(fn.args.args,rb.arg_bindings):
            typ,expr=_binding_expr(binding,f"c_{role}"); args.append((formal.arg,typ,expr))
        nodes=[compiler.sequential.node_by_id[nid] for nid in role_nodes[role]]
        addrs=[typed.node_by_id[n.node_id] for n in nodes]
        out_addr=_output_address_expr(addrs)
        if aby[fid].native_capable: emitters[fid]=_FormulaEmitter(compiler,typed,fid)
        role_meta.append((fid,args,out_addr,nodes))

    family_node_ids={nid for _r,nid in op_sequence}
    dslots,islots,bslots=_initial_typed_pools(compiler,typed,family_node_ids)
    callbacks=[None]*len(plan.formula_ops)
    for fid in set(family.formula_op_ids):
        if not aby[fid].native_capable:
            callbacks[fid]=_python_boundary_callback(compiler,typed,fid,dslots,islots,bslots)

    build_dir=Path(build_dir);build_dir.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha1(("mixed|"+str(family_id)+"|"+"|".join(plan.formula_ops[f].schema_uid for f in family.formula_op_ids)).encode()).hexdigest()[:12]
    module_name=f"{module_prefix}_{digest}"; pyx=build_dir/f"{module_name}.pyx"
    lines=[
        "# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False",
        "import numpy as np", "cimport numpy as cnp",
        "from libc.math cimport pow as c_pow", "",
    ]
    lines += _emit_python_numeric_conversion_helpers()
    for fid in dict.fromkeys(family.formula_op_ids):
        if not aby[fid].native_capable: continue
        role=family.formula_op_ids.index(fid)
        args=[(name,typ) for name,typ,_ in role_meta[role][1]]
        lines.extend(emitters[fid].emit_function(args,aby[fid].observed_return_type.dtype)); lines.append("")
    lines += [
        "cpdef cnp.ndarray run(cnp.ndarray[cnp.float64_t, ndim=1] d_arr, cnp.ndarray[cnp.int64_t, ndim=1] i_arr, cnp.ndarray[cnp.uint8_t, ndim=1] b_arr, object py_callbacks):",
        "    cdef double[::1] dslots = d_arr",
        "    cdef long long[::1] islots = i_arr",
        "    cdef unsigned char[::1] bslots = b_arr",
        f"    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_arr = np.empty({len(op_sequence)}, dtype=np.float64)",
        "    cdef double[::1] out = out_arr",
        "    cdef Py_ssize_t iteration", "    cdef Py_ssize_t out_pos = 0",
    ]
    for role in range(len(family.formula_op_ids)): lines.append(f"    cdef Py_ssize_t c_{role} = 0")
    lines.append(f"    for iteration in range({len(inst.variant_ids)}):")
    python_ops=0
    for role in family.expanded_variants()[0]:
        fid,args,out_addr,_nodes=role_meta[role]
        arg_exprs=[expr for _n,_t,expr in args]
        arr={"double":"dslots","int64":"islots","bool":"bslots"}[out_addr.pool]
        if out_addr.kind=="constant": idx=str(out_addr.a)
        elif out_addr.kind=="affine": idx=f"({out_addr.a} + ({out_addr.b}) * c_{role})"
        else: idx=f"({out_addr.a} + ((c_{role} - ({out_addr.b})) % {out_addr.mod}))"
        if aby[fid].native_capable:
            callargs=", ".join(arg_exprs+["dslots","islots","bslots"])
            value=f"f_{fid}({callargs})"
        else:
            python_ops += len(inst.variant_ids)
            callargs=", ".join(arg_exprs)
            pycall=f"py_callbacks[{fid}]({callargs})" if callargs else f"py_callbacks[{fid}]()"
            dtype=aby[fid].observed_return_type.dtype
            cast={"float64":"double","int64":"long long","bool":"bint"}[dtype]
            value=f"(<{cast}>({pycall}))"
        lines.append(f"        {arr}[{idx}] = {value}")
        lines.append(f"        out[out_pos] = <double>{arr}[{idx}]")
        lines.append("        out_pos += 1"); lines.append(f"        c_{role} += 1")
    lines.append("    return out_arr")
    pyx.write_text("\n".join(lines)+"\n")
    mod,so_path,_proc=build_extension(pyx,module_name,build_dir,openmp=False,optimization="O2",native_arch=False)
    actual=np.asarray(mod.run(dslots,islots,bslots,callbacks),dtype=float)
    byid=compiler.sequential.node_by_id
    expected=np.asarray([float(_authoritative_value(byid[nid])) for _r,nid in op_sequence],dtype=float)
    diff=np.abs(actual-expected); max_abs=float(diff.max()) if diff.size else 0.0
    exact=bool(np.array_equal(actual,expected))
    return MixedNativeFamilyProof(
        family_id,tuple(family.formula_op_ids),
        sum(aby[f].native_capable for f in set(family.formula_op_ids)),
        sum(not aby[f].native_capable for f in set(family.formula_op_ids)),
        len(op_sequence),python_ops,str(pyx),str(so_path),max_abs,exact,
    )

# ---------------------------------------------------------------------------
# Stage 4b.2 foundation: instance-independent family kernel + runtime data
# ---------------------------------------------------------------------------

from .native_bindings import (
    BIND_AFFINE_INT, BIND_ARITH_RUNS_INT, BIND_CONST, BIND_INTERLEAVED_AFFINE_INT,
    BIND_PERIODIC, BIND_RUN_LENGTH, BIND_TABLE, EncodedRuntimeBindings,
    NativeBindingError, RuntimeArgSite, build_runtime_arg_sites, encode_runtime_bindings,
)
from .native_plan import (
    NativeFamilyBackendPlan,
    NativePlanError,
    build_native_family_backend_plan,
    expand_native_family_instance,
)
from .native_references import (
    NativeReferenceError, ReferenceBindingPlan, ReferenceSitePlan,
    _cell_call_name, _impl_from_formula_global, build_reference_binding_plan,
    encode_reference_instance, reference_structural_kinds,
)


@dataclass(frozen=True)
class MixedNativeFamilyInstanceProof:
    instance_ordinal: int
    block_index: int
    operation_count: int
    python_callback_operations: int
    max_abs_error: float
    exact: bool


@dataclass(frozen=True)
class MixedNativeFamilyKernelProof:
    """Proof that one generated family body consumes several runtime instances."""

    family_id: int
    code_signature: str
    formula_op_ids: tuple[int, ...]
    variant_count: int
    instance_count: int
    native_formula_ops: int
    python_formula_ops: int
    pyx_path: str
    so_path: str
    source_lines: int
    source_bytes: int
    instances: tuple[MixedNativeFamilyInstanceProof, ...]

    @property
    def exact(self) -> bool:
        return all(x.exact for x in self.instances)

    @property
    def operation_count(self) -> int:
        return sum(x.operation_count for x in self.instances)

    @property
    def python_callback_operations(self) -> int:
        return sum(x.python_callback_operations for x in self.instances)


_FLOAT_PROOF_MAX_ULP = 8


def _float64_ulp_distance(actual: float, expected: float) -> int | None:
    """Return exact float64 ULP distance, or ``None`` for non-finite mismatch."""
    a = np.float64(actual)
    e = np.float64(expected)
    if np.isnan(a) or np.isnan(e):
        return 0 if np.isnan(a) and np.isnan(e) else None
    if a == e:
        return 0
    if np.isinf(a) or np.isinf(e):
        return None
    ai = int(np.asarray([a], dtype=np.float64).view(np.int64)[0])
    ei = int(np.asarray([e], dtype=np.float64).view(np.int64)[0])
    # Map signed IEEE-754 ordering to a monotonically increasing integer space.
    if ai < 0:
        ai = 0x8000000000000000 - ai
    if ei < 0:
        ei = 0x8000000000000000 - ei
    return abs(ai - ei)


def _hybrid_value_equal(actual: Any, expected: Any) -> bool:
    """Semantic proof equality, allowing at most eight float64 ULPs.

    Integer/bool/string/object structure stays exact.  The small floating-point
    tolerance is fixed up front rather than widened per benchmark; it accepts the
    previously investigated IUL differences (maximum seven ULPs).
    """
    if actual is expected:
        return True
    if (
        isinstance(actual, numbers.Real) and isinstance(expected, numbers.Real)
        and not isinstance(actual, (bool, np.bool_))
        and not isinstance(expected, (bool, np.bool_))
        and (isinstance(actual, (float, np.floating)) or isinstance(expected, (float, np.floating)))
    ):
        distance = _float64_ulp_distance(float(actual), float(expected))
        return distance is not None and distance <= _FLOAT_PROOF_MAX_ULP
    # pandas Series/DataFrame/Index and several other containers expose a scalar
    # ``equals`` method specifically because ``==`` is elementwise.
    equals = getattr(actual, "equals", None)
    if callable(equals) and type(actual) is type(expected):
        try:
            return bool(equals(expected))
        except Exception:
            return False
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(actual), np.asarray(expected)))
        except Exception:
            return False
    try:
        eq = actual == expected
    except Exception:
        return False
    if isinstance(eq, (bool, np.bool_)):
        return bool(eq)
    return False


def _hybrid_numeric_error(actual: list[Any], expected: list[Any]) -> float:
    errors: list[float] = []
    for a, e in zip(actual, expected):
        if (
            isinstance(a, numbers.Number)
            and isinstance(e, numbers.Number)
            and not isinstance(a, (bool, np.bool_))
            and not isinstance(e, (bool, np.bool_))
        ):
            try:
                errors.append(abs(float(a) - float(e)))
            except Exception:
                pass
    return max(errors, default=0.0)


def _runtime_arg_sites(backend: NativeFamilyBackendPlan) -> tuple[RuntimeArgSite, ...]:
    try:
        return build_runtime_arg_sites(backend)
    except NativeBindingError as exc:
        raise NativeFamilyError(str(exc)) from exc


def _site_map(sites: tuple[RuntimeArgSite, ...]) -> dict[tuple[int, int], RuntimeArgSite]:
    return {(x.role, x.arg_index): x for x in sites}


def _encode_runtime_bindings(inst: Any, sites: tuple[RuntimeArgSite, ...]) -> EncodedRuntimeBindings:
    try:
        return encode_runtime_bindings(inst, sites)
    except NativeBindingError as exc:
        raise NativeFamilyError(str(exc)) from exc


def _instance_output_descriptors(
    typed: TypedSlotPlan,
    backend: NativeFamilyBackendPlan,
    expansion: Any,
) -> tuple[np.ndarray, ...]:
    role_count = backend.kernel.role_count
    kind = np.full(role_count, -1, dtype=np.int64)
    a = np.zeros(role_count, dtype=np.int64)
    b = np.zeros(role_count, dtype=np.int64)
    mod = np.zeros(role_count, dtype=np.int64)
    table_offset = np.zeros(role_count, dtype=np.int64)
    table: list[int] = []
    byid = expansion.role_node_ids
    for role in range(role_count):
        node_ids = byid[role]
        if not node_ids:
            continue
        addresses = []
        for nid in node_ids:
            try:
                addresses.append(typed.node_by_id[nid])
            except KeyError as exc:
                raise NativeFamilyError(
                    f"family role {role} output node {nid} has no typed physical address"
                ) from exc
        try:
            expr = _output_address_expr(addresses)
        except NativeFamilyError:
            # Exact address-stream fallback.  This stays runtime data, so irregular
            # liveness layouts do not duplicate generated Cython source.
            kind[role] = 3
            table_offset[role] = len(table)
            table.extend(int(addr.offset) for addr in addresses)
            continue
        kind[role] = {"constant": 0, "affine": 1, "modulo": 2}[expr.kind]
        a[role] = expr.a
        b[role] = expr.b
        mod[role] = expr.mod
    return kind, a, b, mod, table_offset, np.asarray(table, dtype=np.int64)



def _family_output_structural_kinds(
    compiler: RealizedTraceCompiler,
    typed: TypedSlotPlan,
    backend: NativeFamilyBackendPlan,
) -> tuple[int, ...]:
    """Return a compiled output-address kind where all realized instances agree.

    -1 means the role is never realized (normally impossible for executable family
    code); -2 means compatible instances use different address representations and
    the generated body must keep the tiny runtime kind dispatch.
    """
    observed = [set() for _ in range(backend.kernel.role_count)]
    for ordinal in range(len(backend.instances)):
        expansion = expand_native_family_instance(compiler, backend, ordinal)
        kinds = _instance_output_descriptors(typed, backend, expansion)[0]
        for role, kind in enumerate(kinds):
            if int(kind) >= 0:
                observed[role].add(int(kind))
    result = []
    for kinds in observed:
        if not kinds:
            result.append(-1)
        elif len(kinds) == 1:
            result.append(next(iter(kinds)))
        else:
            result.append(-2)
    return tuple(result)


def _emit_output_index_expr(role: int, counter: str, structural_kind: int) -> str:
    if structural_kind == 0:
        return f"out_a[{role}]"
    if structural_kind == 1:
        return f"(out_a[{role}] + out_b[{role}] * {counter})"
    if structural_kind == 2:
        return f"(out_a[{role}] + (({counter} - out_b[{role}]) % out_mod[{role}]))"
    if structural_kind == 3:
        return f"out_table[out_table_offset[{role}] + {counter}]"
    return (
        f"_out_index({role}, {counter}, out_kind, out_a, out_b, out_mod, "
        "out_table_offset, out_table)"
    )


def _runtime_binding_helper_args() -> str:
    return (
        "bind_kind, bind_ia, bind_ib, bind_da, bind_payload_offset, bind_payload_count, "
        "bind_aux_offset, bind_aux_count, bind_i_payload, bind_d_payload, "
        "bind_object_payload, bind_run_ends, bind_cursor"
    )


def _emit_runtime_arg_expr(site: RuntimeArgSite, counter: str) -> str:
    sid = site.site_id
    # Constant and affine representations are sufficiently cheap/common to inline
    # when every instance agrees. All other canonical representations use the
    # shared descriptor ABI, keeping generated source independent of payload size.
    if site.binding_kind == "constant":
        if site.cython_type == "double":
            return f"bind_da[{sid}]"
        if site.cython_type == "bint":
            return f"(bind_ia[{sid}] != 0)"
        if site.cython_type == "long long":
            return f"bind_ia[{sid}]"
        return f"_bind_obj({sid}, {counter}, {_runtime_binding_helper_args()})"
    if site.binding_kind == "affine_int" and site.cython_type == "long long":
        return f"(bind_ia[{sid}] + bind_ib[{sid}] * {counter})"
    helper = {
        "long long": "_bind_i64",
        "double": "_bind_f64",
        "bint": "_bind_bool",
        "object": "_bind_obj",
    }[site.cython_type]
    return f"{helper}({sid}, {counter}, {_runtime_binding_helper_args()})"


def _emit_python_numeric_conversion_helpers() -> list[str]:
    return [
        "from cpython.unicode cimport PyUnicode_Check",
        "from cpython.bytes cimport PyBytes_Check",
        "from cpython.bytearray cimport PyByteArray_Check",
        "cdef extern from \"Python.h\":",
        "    double _mxg_PyFloat_AsDouble \"PyFloat_AsDouble\"(object)",
        "    void* _mxg_PyErr_Occurred \"PyErr_Occurred\"()",
        "    void _mxg_PyErr_Clear \"PyErr_Clear\"()",
        "",
        "cdef inline double _mxg_float(object value):",
        "    cdef double result = _mxg_PyFloat_AsDouble(value)",
        "    if result == -1.0 and _mxg_PyErr_Occurred() != NULL:",
        "        _mxg_PyErr_Clear()",
        "        return <double>float(value)",
        "    return result",
        "",
        "cdef inline long long _mxg_int(object value):",
        "    if PyUnicode_Check(value) or PyBytes_Check(value) or PyByteArray_Check(value):",
        "        return <long long>int(value)",
        "    return <long long>value",
        "",
    ]


def _emit_runtime_helpers() -> list[str]:
    common_sig = (
        "long long[::1] kind, long long[::1] ia, long long[::1] ib, double[::1] da, "
        "long long[::1] payload_offset, long long[::1] payload_count, "
        "long long[::1] aux_offset, long long[::1] aux_count, "
        "long long[::1] i_payload, double[::1] d_payload, object object_payload, "
        "long long[::1] aux_i_payload, long long[::1] cursor"
    )
    lines = _emit_python_numeric_conversion_helpers()
    lines += [
        f"cdef inline Py_ssize_t _binding_pos(Py_ssize_t site, Py_ssize_t occurrence, {common_sig}):",
        "    cdef long long k = kind[site]",
        "    cdef Py_ssize_t off, count, aoff, acount, r, phase, cycle, start_pos",
        f"    if k == {BIND_PERIODIC}:",
        "        count = payload_count[site]",
        "        if count <= 0: raise ValueError('empty periodic runtime binding')",
        "        return occurrence % count",
        f"    if k == {BIND_TABLE}:",
        "        count = payload_count[site]",
        "        if occurrence < 0 or occurrence >= count: raise IndexError('table binding occurrence out of range')",
        "        return occurrence",
        f"    if k == {BIND_RUN_LENGTH}:",
        "        aoff = aux_offset[site]",
        "        acount = aux_count[site]",
        "        r = cursor[site]",
        "        while r < acount and occurrence >= aux_i_payload[aoff + r]:",
        "            r += 1",
        "        if r >= acount: raise IndexError('run-length binding occurrence out of range')",
        "        cursor[site] = r",
        "        return r",
        "    raise ValueError('runtime binding kind has no direct payload position')",
        "",
        f"cdef inline long long _bind_i64(Py_ssize_t site, Py_ssize_t occurrence, {common_sig}):",
        "    cdef long long k = kind[site]",
        "    cdef Py_ssize_t off, count, aoff, acount, r, phase, cycle, start_pos, pos",
        f"    if k == {BIND_CONST}:",
        "        return ia[site]",
        f"    if k == {BIND_AFFINE_INT}:",
        "        return ia[site] + ib[site] * occurrence",
        f"    if k == {BIND_INTERLEAVED_AFFINE_INT}:",
        "        off = payload_offset[site]",
        "        count = payload_count[site]",
        "        aoff = aux_offset[site]",
        "        if count <= 0 or aux_count[site] != count: raise ValueError('bad interleaved binding payload')",
        "        phase = occurrence % count",
        "        cycle = occurrence // count",
        "        return i_payload[off + phase] + aux_i_payload[aoff + phase] * cycle",
        f"    if k == {BIND_ARITH_RUNS_INT}:",
        "        off = payload_offset[site]",
        "        aoff = aux_offset[site]",
        "        acount = aux_count[site]",
        "        r = cursor[site]",
        "        while r < acount and occurrence >= aux_i_payload[aoff + r]:",
        "            r += 1",
        "        if r >= acount: raise IndexError('arithmetic-run binding occurrence out of range')",
        "        cursor[site] = r",
        "        start_pos = 0 if r == 0 else aux_i_payload[aoff + r - 1]",
        "        return i_payload[off + r] + ia[site] * (occurrence - start_pos)",
        f"    if k == {BIND_PERIODIC} or k == {BIND_TABLE} or k == {BIND_RUN_LENGTH}:",
        "        off = payload_offset[site]",
        "        pos = _binding_pos(site, occurrence, kind, ia, ib, da, payload_offset, payload_count, aux_offset, aux_count, i_payload, d_payload, object_payload, aux_i_payload, cursor)",
        "        return i_payload[off + pos]",
        "    raise ValueError('unsupported integer runtime binding kind')",
        "",
        f"cdef inline double _bind_f64(Py_ssize_t site, Py_ssize_t occurrence, {common_sig}):",
        "    cdef long long k = kind[site]",
        "    cdef Py_ssize_t off, pos",
        f"    if k == {BIND_CONST}:",
        "        return da[site]",
        f"    if k == {BIND_PERIODIC} or k == {BIND_TABLE} or k == {BIND_RUN_LENGTH}:",
        "        off = payload_offset[site]",
        "        pos = _binding_pos(site, occurrence, kind, ia, ib, da, payload_offset, payload_count, aux_offset, aux_count, i_payload, d_payload, object_payload, aux_i_payload, cursor)",
        "        return d_payload[off + pos]",
        "    raise ValueError('unsupported double runtime binding kind')",
        "",
        f"cdef inline bint _bind_bool(Py_ssize_t site, Py_ssize_t occurrence, {common_sig}):",
        "    cdef long long k = kind[site]",
        "    cdef Py_ssize_t off, pos",
        f"    if k == {BIND_CONST}:",
        "        return ia[site] != 0",
        f"    if k == {BIND_PERIODIC} or k == {BIND_TABLE} or k == {BIND_RUN_LENGTH}:",
        "        off = payload_offset[site]",
        "        pos = _binding_pos(site, occurrence, kind, ia, ib, da, payload_offset, payload_count, aux_offset, aux_count, i_payload, d_payload, object_payload, aux_i_payload, cursor)",
        "        return i_payload[off + pos] != 0",
        "    raise ValueError('unsupported bool runtime binding kind')",
        "",
        f"cdef inline object _bind_obj(Py_ssize_t site, Py_ssize_t occurrence, {common_sig}):",
        "    cdef long long k = kind[site]",
        "    cdef Py_ssize_t off, pos",
        f"    if k == {BIND_CONST}:",
        "        return object_payload[payload_offset[site]]",
        f"    if k == {BIND_AFFINE_INT} or k == {BIND_INTERLEAVED_AFFINE_INT} or k == {BIND_ARITH_RUNS_INT}:",
        "        return _bind_i64(site, occurrence, kind, ia, ib, da, payload_offset, payload_count, aux_offset, aux_count, i_payload, d_payload, object_payload, aux_i_payload, cursor)",
        f"    if k == {BIND_PERIODIC} or k == {BIND_TABLE} or k == {BIND_RUN_LENGTH}:",
        "        off = payload_offset[site]",
        "        pos = _binding_pos(site, occurrence, kind, ia, ib, da, payload_offset, payload_count, aux_offset, aux_count, i_payload, d_payload, object_payload, aux_i_payload, cursor)",
        "        return object_payload[off + pos]",
        "    raise ValueError('unsupported object runtime binding kind')",
        "",
        "cdef inline Py_ssize_t _out_index(Py_ssize_t role, Py_ssize_t occurrence,",
        "                                  long long[::1] out_kind, long long[::1] out_a,",
        "                                  long long[::1] out_b, long long[::1] out_mod,",
        "                                  long long[::1] out_table_offset, long long[::1] out_table):",
        "    cdef long long k = out_kind[role]",
        "    if k == 0:",
        "        return out_a[role]",
        "    if k == 1:",
        "        return out_a[role] + out_b[role] * occurrence",
        "    if k == 2:",
        "        return out_a[role] + ((occurrence - out_b[role]) % out_mod[role])",
        "    if k == 3:",
        "        return out_table[out_table_offset[role] + occurrence]",
        "    raise ValueError('missing output address descriptor for executed role')",
        "",
    ]
    return lines



RuntimeReferenceSite = ReferenceSitePlan


@dataclass(frozen=True)
class _RuntimeReferenceAddress:
    pool: str
    site_id: int
    occurrence_addressed: bool = False
    structural_kind: int = -2

    def emit(self, arg: str | None = None) -> str:
        arr = {"double": "dslots", "int64": "islots", "bool": "bslots", "object": "oslots"}[self.pool]
        key = "formula_occ" if self.occurrence_addressed else ("0" if arg is None else arg)
        sid = self.site_id
        # Freeze only the descriptor *shape*.  Every concrete address parameter
        # remains per-instance runtime data, preserving the reference/storage ABI.
        if self.structural_kind == 0:
            idx = f"ref_a[{sid}]"
        elif self.structural_kind == 1:
            idx = f"(ref_a[{sid}] + ref_b[{sid}] * ({key}))"
        elif self.structural_kind == 2:
            idx = f"(ref_a[{sid}] + ((({key}) - ref_b[{sid}]) % ref_mod[{sid}]))"
        elif self.structural_kind == 3:
            idx = f"_ref_table_index({sid}, {key}, ref_lo, ref_table_offset, ref_table)"
        elif self.structural_kind == 4:
            idx = f"_ref_occ_index({sid}, {key}, ref_b, ref_table_offset, ref_table)"
        else:
            idx = (
                f"_ref_index({sid}, {key}, ref_kind, ref_a, ref_b, ref_mod, "
                "ref_lo, ref_table_offset, ref_table)"
            )
        expr = f"{arr}[{idx}]"
        if self.pool == "bool":
            return f"({expr} != 0)"
        return expr


class _RuntimeFormulaEmitter(_FormulaEmitter):
    """Formula emitter whose scheduled Cell addresses are runtime descriptors.

    Reference-site identity is derived from FormulaOp source/ABI. Concrete Cells
    objects and physical address payloads are deliberately absent here and are
    supplied per instance by ``ReferenceBindingPlan``.
    """

    def __init__(
        self,
        compiler: RealizedTraceCompiler,
        typed: TypedSlotPlan,
        formula_op_id: int,
        sites: tuple[RuntimeReferenceSite, ...],
        reference_structural_kinds: tuple[int, ...] | None = None,
    ):
        self.compiler = compiler
        self.typed = typed
        self.formula_op_id = formula_op_id
        plan = compiler.structured.canonical_plan
        assert plan is not None
        self.formula_op = plan.formula_ops[formula_op_id]
        reps = [n for n in compiler.trace.nodes if n.shape_token == self.formula_op.role]
        if not reps:
            raise NativeFamilyError(f"no representative node for FormulaOp {formula_op_id}")
        self.rep = reps[0]
        self.fn = _parse_func(self.rep.schema.source)
        # Scalar/non-Cell globals remain representative-environment evidence for
        # now. Direct scheduled Cell calls are no longer resolved from this dict.
        self.globals = self.rep.obj.altfunc.__globals__
        self.formals = {a.arg for a in self.fn.args.args}
        self.locals = set()
        for n in ast.walk(self.fn):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        self.locals.add(target.id)
        self.resolvers = {}
        self.callsite_resolvers = {}
        structural = reference_structural_kinds
        if structural is not None and len(structural) != len(sites):
            raise NativeFamilyError("reference structural-kind vector does not match sites")
        for site in sites:
            if site.formula_op_id != formula_op_id:
                continue
            structural_kind = -2 if structural is None else int(structural[site.site_id])
            resolver = _CallResolver(
                site.name,
                None,
                _RuntimeReferenceAddress(
                    site.pool, site.site_id, site.occurrence_addressed, structural_kind
                ),
                site.arg_count,
            )
            if site.call_lineno >= 0:
                self.callsite_resolvers[(site.name, site.call_lineno, site.call_col_offset)] = resolver
            else:
                self.resolvers[site.name] = resolver

    def emit_function(self, arg_types: list[tuple[str, str]], return_dtype: str) -> list[str]:
        self.formal_cython_types = dict(arg_types)
        rettype = {"float64": "double", "int64": "long long", "bool": "bint"}[return_dtype]
        args = ", ".join(f"{typ} {name}" for name, typ in arg_types)
        if args:
            args += ", "
        lines = [
            f"cdef inline {rettype} f_{self.formula_op_id}(Py_ssize_t formula_occ, {args}"
            "double[::1] dslots, long long[::1] islots, unsigned char[::1] bslots, object oslots, "
            "long long[::1] ref_kind, long long[::1] ref_a, long long[::1] ref_b, "
            "long long[::1] ref_mod, long long[::1] ref_lo, "
            "long long[::1] ref_table_offset, long long[::1] ref_table):"
        ]
        for name in sorted(self.locals - self.formals):
            lines.append(f"    cdef double {name} = 0.0")
        lines.extend(self.statements(self.fn.body, "    "))
        return lines



class ModelxCythonRuntimeFormulaEmitter(_RuntimeFormulaEmitter):
    """Re-emit modelx-cython's traced private formula against graph slots."""

    def expr(self, node: ast.AST) -> str:
        # A qualified Cell read such as ``data.some_cell(t)`` has exactly the
        # same compiler-owned reference ABI as a direct ``some_cell(t)`` read.
        # Resolve it before generic Attribute emission so generated code never
        # retains or calls through a modelx Space object.
        if isinstance(node, ast.Call):
            name = _cell_call_name(node)
            if name is not None and "." in name:
                resolved = self._resolved_cell_call_expr(node, name)
                if resolved is not None:
                    return resolved
                object_globals = getattr(self, "object_globals", {})
                if name in object_globals and not node.args and not node.keywords:
                    return f"_mxg_engine_globals[{int(object_globals[name])}]"
        # modelx-cython deliberately leaves pandas and whole-array NumPy values
        # as Python objects. Preserve that ABI here. The legacy scalar emitter
        # forces both operands of true division to ``double``; doing so inside an
        # object-returning formula tries to coerce Series/ndarray operands to a
        # scalar and destroys vector semantics.
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and self.engine_formula.return_type in {"object", "str"}
        ):
            # The comprehension induction variable in an object formula is often
            # inferred as a C integer by Cython.  Emitting bare ``1 / 12`` would
            # consequently change Python's true-division result to integer zero
            # (a monthly discount-rate vector is the minimal real-world example).
            # At the same time, forcing both sides to ``double`` is not
            # valid for pandas/NumPy object operations.  Box both operands: this
            # preserves Python true-division for scalars and delegates vector
            # semantics to the owning object without crossing a formula callback.
            return (
                f"((<object>({self.expr(node.left)})) / "
                f"(<object>({self.expr(node.right)})))"
            )
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Pow)
            and self.engine_formula.return_type in {"object", "str"}
        ):
            return (
                f"((<object>({self.expr(node.left)})) ** "
                f"(<object>({self.expr(node.right)})))"
            )
        return super().expr(node)

    def __init__(
        self, compiler, typed, formula_op_id, sites, engine_formula, object_globals=None,
        reference_structural_kinds=None,
    ):
        super().__init__(
            compiler, typed, formula_op_id, sites,
            reference_structural_kinds=reference_structural_kinds,
        )
        self.engine_formula = engine_formula
        self.object_globals = dict(object_globals or {})
        self.fn = engine_formula.function
        self.formals = {a.arg for a in self.fn.args.args}
        self.locals = set()

        def add_target(target):
            if isinstance(target, ast.Name):
                self.locals.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for item in target.elts:
                    add_target(item)

        for node in ast.walk(self.fn):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    add_target(target)
            elif isinstance(node, (ast.For, ast.comprehension)):
                add_target(node.target)

    def emit_function(self, arg_types: list[tuple[str, str]], return_dtype: str) -> list[str]:
        self.formal_cython_types = dict(arg_types)
        return_type = self.engine_formula.return_type
        if return_type not in {"double", "long long", "bint", "str", "object"}:
            raise NativeFamilyError(
                f"modelx-cython formula {self.engine_formula.key} has non-scalar return ABI {return_type}"
            )
        args = ", ".join(f"{typ} {name}" for name, typ in arg_types)
        if args:
            args += ", "
        realized_return_type = {
            "float64": "double", "int64": "long long", "bool": "bint"
        }.get(return_dtype, return_type)
        lines = [
            f"cdef inline {realized_return_type} f_{self.formula_op_id}(Py_ssize_t formula_occ, {args}"
            "double[::1] dslots, long long[::1] islots, unsigned char[::1] bslots, object oslots, "
            "long long[::1] ref_kind, long long[::1] ref_a, long long[::1] ref_b, "
            "long long[::1] ref_mod, long long[::1] ref_lo, "
            "long long[::1] ref_table_offset, long long[::1] ref_table):"
        ]
        # Deliberately let Cython infer local variable types from modelx-cython's
        # transformed body.  The legacy emitter's blanket double declaration is
        # incorrect for string/object temporaries and defeats the frontend reuse.
        body = self.statements(self.fn.body, "    ")
        if realized_return_type != return_type:
            converted = []
            for line in body:
                prefix, marker, expression = line.partition("return ")
                if marker:
                    line = prefix + f"return <{realized_return_type}>({expression})"
                converted.append(line)
            body = converted
        lines.extend(body)
        return lines


class _LiteralConstantSpecializer(ast.NodeTransformer):
    """Fold branches proven by one concrete LiteralBlock invocation."""

    def __init__(self, constants: dict[str, Any]):
        self.constants = constants

    def visit_Name(self, node):  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id in self.constants:
            value = self.constants[node.id]
            if isinstance(value, np.generic):
                value = value.item()
            return ast.copy_location(ast.Constant(value), node)
        return node

    def _visit_statements(self, rows):
        out = []
        for statement in rows:
            rewritten = self.visit(statement)
            if rewritten is None:
                continue
            if isinstance(rewritten, list):
                out.extend(rewritten)
            else:
                out.append(rewritten)
        return out

    @staticmethod
    def _static_condition(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (bool, np.bool_)):
            return bool(node.value)
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and len(node.comparators) == 1
            and isinstance(node.left, ast.Constant)
            and isinstance(node.comparators[0], ast.Constant)
        ):
            left, right = node.left.value, node.comparators[0].value
            op = node.ops[0]
            try:
                if isinstance(op, ast.Eq): return left == right
                if isinstance(op, ast.NotEq): return left != right
                if isinstance(op, ast.Lt): return left < right
                if isinstance(op, ast.LtE): return left <= right
                if isinstance(op, ast.Gt): return left > right
                if isinstance(op, ast.GtE): return left >= right
                if isinstance(op, ast.Is): return left is right
                if isinstance(op, ast.IsNot): return left is not right
            except Exception:
                return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = _LiteralConstantSpecializer._static_condition(node.operand)
            return None if value is None else not value
        return None

    def visit_If(self, node):  # noqa: N802
        test = self.visit(node.test)
        value = self._static_condition(test)
        if isinstance(value, (bool, np.bool_)):
            return self._visit_statements(node.body if bool(value) else node.orelse)
        node.test = test
        node.body = self._visit_statements(node.body)
        node.orelse = self._visit_statements(node.orelse)
        return node


class LiteralModelxCythonFormulaEmitter(ModelxCythonRuntimeFormulaEmitter):
    """Re-emit one concrete LiteralBlock formula against direct graph slots.

    LiteralBlock operations occur only a handful of times and are outside recovered
    loop families.  Their exact dependency edges are already known, so there is no
    reason to route each Cell read through the family reference-descriptor ABI.
    This keeps the same modelx-cython-derived formula body while binding Cell calls
    directly to the compiler-owned physical slots for this concrete occurrence.
    """

    def __init__(
        self, compiler, typed, formula_op_id, node, engine_formula,
        object_globals=None, reference_pool_hints=None,
    ):
        # Reuse formula normalization/object-global semantics, then replace the
        # family runtime-descriptor resolvers with exact concrete-slot resolvers.
        super().__init__(compiler, typed, formula_op_id, (), engine_formula, object_globals=object_globals)
        self.rep = node
        self.output_pool = typed.node_by_id[node.node_id].pool
        self.globals = node.obj.altfunc.__globals__
        constants = {
            name: value
            for (name, _typ), value in zip(engine_formula.arg_types, node.args)
            if isinstance(value, (bool, int, float, str, np.bool_, np.integer, np.floating))
        }
        if constants:
            specialized = _LiteralConstantSpecializer(constants).visit(ast.fix_missing_locations(ast.parse(ast.unparse(self.fn))).body[0])
            if not isinstance(specialized, ast.FunctionDef):
                raise NativeFamilyError("literal formula specialization did not preserve a function")
            self.fn = ast.fix_missing_locations(specialized)
        self.resolvers = {}
        self.callsite_resolvers = {}
        reference_pool_hints = dict(reference_pool_hints or {})
        byid = compiler.sequential.node_by_id
        incoming = [src for src, dst in compiler.trace.dependencies if dst == node.node_id]
        direct_dependencies = [byid[src] for src in incoming if src in typed.node_by_id]
        for name in {
            call_name for call in ast.walk(self.fn)
            if isinstance(call, ast.Call)
            for call_name in [_cell_call_name(call)]
            if call_name is not None
        }:
            calls = sorted(
                (
                    call for call in ast.walk(self.fn)
                    if isinstance(call, ast.Call)
                    and _cell_call_name(call) == name
                ),
                key=lambda call: (
                    int(getattr(call, "lineno", -1)),
                    int(getattr(call, "col_offset", -1)),
                ),
            )
            impl = _impl_from_formula_global(
                self.globals, name, calls[0] if calls else None
            )
            if impl is None:
                continue
            deps = [dep for dep in direct_dependencies if dep.obj is impl]
            for call in calls:
                argc = len(call.args)
                # modelx records defaulted trailing arguments in the dependency
                # key even when formula source supplied only the leading values.
                # Project those concrete keys to the source-visible arity.
                call_deps = [dep for dep in deps if len(dep.args) >= argc]
                if call_deps:
                    addresses = [typed.node_by_id[dep.node_id] for dep in call_deps]
                    projected = [
                        _ProjectedCallNode(tuple(dep.args[:argc])) for dep in call_deps
                    ]
                    address = _fit_literal_address(projected, addresses)
                else:
                    pools = {
                        typed.node_by_id[candidate.node_id].pool
                        for candidate in compiler.trace.nodes
                        if candidate.obj is impl
                        and len(candidate.args) == argc
                        and candidate.node_id in typed.node_by_id
                    }
                    hinted = reference_pool_hints.get(name)
                    if hinted in {"double", "int64", "bool", "object"}:
                        pools.add(hinted)
                    # A cached Cell with no graph node was not reached by any
                    # realized occurrence, so this is a proven dormant branch. An
                    # uncached helper may execute without producing a scheduled
                    # node and must therefore remain a Python island until helper
                    # inlining exists.
                    if not pools:
                        if getattr(impl, "is_cached", False):
                            pools.add("object")
                        else:
                            raise NativeFamilyError(
                                f"literal uncached helper {name} has no scheduled dependency"
                            )
                    if len(pools) != 1:
                        raise NativeFamilyError(
                            f"literal Cell call {name} has no unique typed dependency"
                        )
                    address = _ConcreteLiteralAddressExpr(next(iter(pools)), ())
                resolver = _CallResolver(name, impl, address, argc)
                if len(calls) == 1:
                    self.resolvers[name] = resolver
                else:
                    self.callsite_resolvers[(
                        name,
                        int(getattr(call, "lineno", -1)),
                        int(getattr(call, "col_offset", -1)),
                    )] = resolver

    def emit_literal_function(self, function_name: str, arg_types: list[tuple[str, str]]) -> list[str]:
        self.formal_cython_types = dict(arg_types)
        return_type = self.engine_formula.return_type
        if return_type not in {"double", "long long", "bint", "str", "object"}:
            raise NativeFamilyError(
                f"literal formula {self.engine_formula.key} has unsupported return ABI {return_type}"
            )
        args = ", ".join(f"{typ} {name}" for name, typ in arg_types)
        if args:
            args += ", "
        realized_return_type = {
            "double": "double", "int64": "long long", "bool": "bint", "object": "object"
        }.get(self.output_pool, return_type)
        lines = [
            f"cdef inline {realized_return_type} {function_name}({args}"
            "double[::1] dslots, long long[::1] islots, unsigned char[::1] bslots, object oslots):"
        ]
        body = self.statements(self.fn.body, "    ")
        if realized_return_type != return_type:
            converted = []
            for line in body:
                prefix, marker, expression = line.partition("return ")
                if marker:
                    line = prefix + f"return <{realized_return_type}>({expression})"
                converted.append(line)
            body = converted
        lines.extend(body)
        return lines


def modelx_cython_formula_native_ok(engine_formula, output_pool: str) -> bool:
    if engine_formula is None:
        return False
    # Keep the reuse boundary semantic rather than driven by a performance cost
    # model. modelx-cython itself keeps pandas and whole-array NumPy values as
    # Python objects (apart from the narrow element-only ndarray memoryview case).
    # modelx-cython compiles object-returning private formulas as cfuncs too.
    # Keeping object operations in Cython preserves pandas/NumPy islands while
    # removing the Python formula-call boundary; it does not scalarize them.
    expected = {
        "double": {"double"},
        # A traced integer slot can be a sentinel-only realization of a formula
        # with dormant floating branches.  The generated role helper performs
        # the same narrowing conversion as assignment to the realized slot.
        "int64": {"long long", "double"},
        "bool": {"bint"},
        "object": {"str", "object"},
    }.get(output_pool, set())
    return engine_formula.return_type in expected

def _runtime_reference_plan(
    compiler: RealizedTraceCompiler,
    typed: TypedSlotPlan,
    backend: NativeFamilyBackendPlan,
    native_by_id: dict[int, bool],
) -> ReferenceBindingPlan:
    try:
        return build_reference_binding_plan(compiler, typed, backend, native_by_id)
    except NativeReferenceError as exc:
        raise NativeFamilyError(str(exc)) from exc


def _emit_runtime_reference_helper() -> list[str]:
    return [
        "cdef inline Py_ssize_t _ref_table_index(Py_ssize_t site, long long key,",
        "                                        long long[::1] lo, long long[::1] table_offset,",
        "                                        long long[::1] table):",
        "    cdef Py_ssize_t pos = table_offset[site] + (key - lo[site])",
        "    if key < lo[site] or pos < 0 or pos >= table.shape[0] or table[pos] < 0:",
        "        raise KeyError('runtime Cell reference key outside realized address table')",
        "    return table[pos]",
        "",
        "cdef inline Py_ssize_t _ref_occ_index(Py_ssize_t site, long long key,",
        "                                      long long[::1] count, long long[::1] table_offset,",
        "                                      long long[::1] table):",
        "    cdef Py_ssize_t pos",
        "    if key < 0 or key >= count[site]: raise IndexError('formula occurrence outside realized reference stream')",
        "    pos = table_offset[site] + key",
        "    if pos < 0 or pos >= table.shape[0] or table[pos] < 0:",
        "        raise KeyError('dormant Cell reference executed outside realized dependency stream')",
        "    return table[pos]",
        "",
        "cdef inline Py_ssize_t _ref_index(Py_ssize_t site, long long key,",
        "                                  long long[::1] kind, long long[::1] a,",
        "                                  long long[::1] b, long long[::1] mod,",
        "                                  long long[::1] lo, long long[::1] table_offset,",
        "                                  long long[::1] table):",
        "    cdef long long k = kind[site]",
        "    if k == 0:",
        "        return a[site]",
        "    if k == 1:",
        "        return a[site] + b[site] * key",
        "    if k == 2:",
        "        return a[site] + ((key - b[site]) % mod[site])",
        "    if k == 3:",
        "        return _ref_table_index(site, key, lo, table_offset, table)",
        "    if k == 4:",
        "        return _ref_occ_index(site, key, b, table_offset, table)",
        "    raise ValueError('unsupported runtime reference address kind')",
        "",
    ]



def _exact_native_formula_ok(
    compiler: RealizedTraceCompiler,
    formula_op_id: int,
    analysis_row: Any,
    *,
    allow_pow: bool = False,
) -> bool:
    """Conservative exactness gate for the Stage 4b.2 foundation kernel.

    CPython float power and C/Cython ``pow`` can differ by a few ulps even when
    expression order is unchanged.  Until the native backend has an explicit
    bitwise-compatible power implementation, formulas containing ``**`` remain
    Python callbacks in this proof path.  This is a semantic guard, not a model
    special case.
    """
    if not analysis_row.native_capable:
        return False
    plan = compiler.structured.canonical_plan
    assert plan is not None
    op = plan.formula_ops[formula_op_id]
    rep = next((n for n in compiler.trace.nodes if n.shape_token == op.role), None)
    if rep is None:
        return False
    try:
        fn = _parse_func(rep.schema.source)
    except Exception:
        return False
    if (not allow_pow) and any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) for n in ast.walk(fn)):
        return False
    # The current emitter gives inferred local temporaries C double storage.
    # That is exact for the float-heavy foundation slice, but it is not a valid
    # contract for integer/bool-return formulas whose local temporaries may be
    # logically integral. Until local SSA/type inference exists, keep those
    # formulas as Python islands rather than permit an implicit C conversion.
    if analysis_row.observed_return_type.dtype != "float64":
        for n in ast.walk(fn):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(isinstance(target, ast.Name) for target in targets):
                    return False
    return True


def build_mixed_native_family_kernel_cython(
    compiler: RealizedTraceCompiler,
    family_id: int,
    build_dir: str | Path,
    *,
    instance_ordinals: tuple[int, ...] | None = None,
    module_prefix: str = "mxg_family_kernel",
) -> MixedNativeFamilyKernelProof:
    """Compile one shared multi-variant family kernel and execute selected instances.

    This is the Stage 4b.2 foundation proof.  Formula/variant syntax is emitted
    once.  Exact variant streams, formal-argument bindings and output-slot address
    streams are supplied as per-instance runtime data.  Python FormulaOps remain
    callbacks over the same slot pools, including a list-backed object pool for
    arbitrary Python state.  Native numeric FormulaOps continue to use zero-copy
    typed memoryviews and never reinterpret object slots.
    """

    if compiler.slots is None:
        compiler.lower_slots()
    analysis = compiler.native_plan or compiler.analyze_native()
    typed = _typed_slot_plan(compiler, analysis, include_objects=True)
    try:
        backend = build_native_family_backend_plan(compiler, family_id)
    except NativePlanError as exc:
        raise NativeFamilyError(str(exc)) from exc
    plan = compiler.structured.canonical_plan
    assert plan is not None
    aby = {x.formula_op_id: x for x in analysis.formulae}

    formula_ids = tuple(backend.kernel.formula_op_ids)
    for fid in set(formula_ids):
        if fid not in typed.formula_by_id:
            raise NativeFamilyError(f"FormulaOp {fid} has no physical slot allocation")

    sites = _runtime_arg_sites(backend)
    kernel_native = {
        fid: _exact_native_formula_ok(compiler, fid, aby[fid]) for fid in set(formula_ids)
    }
    # Preserve exact Python argument categories. If any role using a FormulaOp has
    # an object-valued/heterogeneous formal site, that FormulaOp remains Python.
    for role, fid in enumerate(formula_ids):
        if any(site.cython_type == "object" for site in sites if site.role == role):
            kernel_native[fid] = False
    reference_plan = _runtime_reference_plan(compiler, typed, backend, kernel_native)
    for fid in reference_plan.unsupported_formula_ops:
        kernel_native[fid] = False
    reference_sites = reference_plan.sites
    reference_kinds = reference_structural_kinds(reference_plan)
    output_structural_kinds = _family_output_structural_kinds(compiler, typed, backend)
    emitters: dict[int, _RuntimeFormulaEmitter] = {}
    role_meta: list[tuple[int, list[tuple[str, str, RuntimeArgSite]], str]] = []

    for role, fid in enumerate(formula_ids):
        op = plan.formula_ops[fid]
        site_rows = sorted((s for s in sites if s.role == role), key=lambda x: x.arg_index)
        if len(site_rows) != op.arity:
            raise NativeFamilyError("FormulaOp arity does not match runtime binding sites")
        if kernel_native[fid]:
            emitter = emitters.setdefault(
                fid, _RuntimeFormulaEmitter(
                    compiler, typed, fid, reference_sites,
                    reference_structural_kinds=reference_kinds,
                )
            )
            formal_names = [a.arg for a in emitter.fn.args.args]
            if len(formal_names) != op.arity:
                raise NativeFamilyError("native formula formal arity does not match FormulaOp")
        else:
            # Python callbacks do not need source parsing here.  Some legitimate
            # modelx formulas have source forms that the conservative native parser
            # intentionally marks unparseable.
            formal_names = [f"arg{i}" for i in range(op.arity)]
        args = [(name, site.cython_type, site) for name, site in zip(formal_names, site_rows)]
        role_meta.append((fid, args, typed.formula_by_id[fid].pool))

    has_object_outputs = any(pool == "object" for _fid, _args, pool in role_meta)

    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    module_name = f"{module_prefix}_{backend.kernel.code_signature[:12]}"
    pyx = build_dir / f"{module_name}.pyx"
    lines = [
        "# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False",
        "import numpy as np",
        "cimport numpy as cnp",
        "from libc.math cimport pow as c_pow",
        "",
    ]
    lines.extend(_emit_runtime_helpers())
    lines.extend(_emit_runtime_reference_helper())

    for fid in dict.fromkeys(formula_ids):
        if not kernel_native[fid]:
            continue
        role = formula_ids.index(fid)
        args = [(name, typ) for name, typ, _site in role_meta[role][1]]
        lines.extend(emitters[fid].emit_function(args, aby[fid].observed_return_type.dtype))
        lines.append("")

    lines += [
        "cpdef object run(",
        "    cnp.ndarray[cnp.float64_t, ndim=1] d_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] i_arr,",
        "    cnp.ndarray[cnp.uint8_t, ndim=1] b_arr,",
        "    object o_slots,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] variant_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] bind_kind_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] bind_ia_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] bind_ib_arr,",
        "    cnp.ndarray[cnp.float64_t, ndim=1] bind_da_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] bind_payload_offset_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] bind_payload_count_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] bind_aux_offset_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] bind_aux_count_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] bind_i_payload_arr,",
        "    cnp.ndarray[cnp.float64_t, ndim=1] bind_d_payload_arr,",
        "    object bind_object_payload,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] bind_run_ends_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] bind_cursor_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] ref_kind_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] ref_a_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] ref_b_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] ref_mod_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] ref_lo_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] ref_table_offset_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] ref_table_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] out_kind_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] out_a_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] out_b_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] out_mod_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] out_table_offset_arr,",
        "    cnp.ndarray[cnp.int64_t, ndim=1] out_table_arr,",
        "    object py_callbacks,",
        "    Py_ssize_t out_count):",
        "    cdef double[::1] dslots = d_arr",
        "    cdef long long[::1] islots = i_arr",
        "    cdef unsigned char[::1] bslots = b_arr",
        "    cdef long long[::1] variants = variant_arr",
        "    cdef long long[::1] bind_kind = bind_kind_arr",
        "    cdef long long[::1] bind_ia = bind_ia_arr",
        "    cdef long long[::1] bind_ib = bind_ib_arr",
        "    cdef double[::1] bind_da = bind_da_arr",
        "    cdef long long[::1] bind_payload_offset = bind_payload_offset_arr",
        "    cdef long long[::1] bind_payload_count = bind_payload_count_arr",
        "    cdef long long[::1] bind_aux_offset = bind_aux_offset_arr",
        "    cdef long long[::1] bind_aux_count = bind_aux_count_arr",
        "    cdef long long[::1] bind_i_payload = bind_i_payload_arr",
        "    cdef double[::1] bind_d_payload = bind_d_payload_arr",
        "    cdef long long[::1] bind_run_ends = bind_run_ends_arr",
        "    cdef long long[::1] bind_cursor = bind_cursor_arr",
        "    cdef long long[::1] ref_kind = ref_kind_arr",
        "    cdef long long[::1] ref_a = ref_a_arr",
        "    cdef long long[::1] ref_b = ref_b_arr",
        "    cdef long long[::1] ref_mod = ref_mod_arr",
        "    cdef long long[::1] ref_lo = ref_lo_arr",
        "    cdef long long[::1] ref_table_offset = ref_table_offset_arr",
        "    cdef long long[::1] ref_table = ref_table_arr",
        "    cdef long long[::1] out_kind = out_kind_arr",
        "    cdef long long[::1] out_a = out_a_arr",
        "    cdef long long[::1] out_b = out_b_arr",
        "    cdef long long[::1] out_mod = out_mod_arr",
        "    cdef long long[::1] out_table_offset = out_table_offset_arr",
        "    cdef long long[::1] out_table = out_table_arr",
        "    cdef Py_ssize_t iteration, variant, out_pos = 0, idx",
    ]
    if has_object_outputs:
        lines.append("    cdef object result_obj = [None] * out_count")
    else:
        lines += [
            "    cdef cnp.ndarray[cnp.float64_t, ndim=1] result_arr = np.empty(out_count, dtype=np.float64)",
            "    cdef double[::1] result = result_arr",
        ]
    for role in range(len(formula_ids)):
        lines.append(f"    cdef Py_ssize_t c_{role} = 0")
    lines.append("    for iteration in range(variants.shape[0]):")
    lines.append("        variant = variants[iteration]")

    for variant_id, pattern in enumerate(backend.kernel.variants):
        prefix = "if" if variant_id == 0 else "elif"
        lines.append(f"        {prefix} variant == {variant_id}:")
        if not pattern:
            lines.append("            pass")
            continue
        for role in pattern:
            fid, args, pool = role_meta[role]
            arg_exprs = [_emit_runtime_arg_expr(site, f"c_{role}") for _name, _typ, site in args]
            if kernel_native[fid]:
                # Runtime formula functions are occurrence-aware and may read
                # numeric or object physical slots.  Keep this foundation caller
                # aligned with the shared formula ABI used by the whole-artifact
                # runner: occurrence first, then formal arguments and all slot /
                # reference pools.  The previous caller predated occurrence-
                # addressed references and silently omitted both ``formula_occ``
                # and ``oslots``; medium source-expanded families exposed the
                # mismatch at Cython compile time.
                callargs = ", ".join(
                    [f"c_{role}"]
                    + arg_exprs
                    + [
                        "dslots", "islots", "bslots", "o_slots",
                        "ref_kind", "ref_a", "ref_b", "ref_mod",
                        "ref_lo", "ref_table_offset", "ref_table",
                    ]
                )
                value = f"f_{fid}({callargs})"
            else:
                pyargs = ", ".join(arg_exprs)
                pycall = (
                    f"py_callbacks[{role}][c_{role}]({pyargs})"
                    if pyargs else f"py_callbacks[{role}][c_{role}]()"
                )
                dtype = aby[fid].observed_return_type.dtype
                if dtype == "object":
                    value = pycall
                else:
                    cast = {"float64": "double", "int64": "long long", "bool": "bint"}[dtype]
                    value = f"(<{cast}>({pycall}))"
            arr = {
                "double": "dslots", "int64": "islots", "bool": "bslots", "object": "o_slots"
            }[pool]
            lines.append(
                f"            idx = {_emit_output_index_expr(role, f'c_{role}', output_structural_kinds[role])}"
            )
            lines.append(f"            {arr}[idx] = {value}")
            if has_object_outputs:
                lines.append(f"            result_obj[out_pos] = {arr}[idx]")
            else:
                lines.append(f"            result[out_pos] = <double>{arr}[idx]")
            lines.append("            out_pos += 1")
            lines.append(f"            c_{role} += 1")
    lines.append("        else:")
    lines.append("            raise ValueError('variant id outside compiled family grammar')")
    lines.append("    if out_pos != out_count:")
    lines.append("        raise ValueError('family output count does not match runtime variant stream')")
    lines.append("    return result_obj" if has_object_outputs else "    return result_arr")

    source = "\n".join(lines) + "\n"
    pyx.write_text(source)
    mod, so_path, _proc = build_extension(
        pyx, module_name, build_dir, openmp=False, optimization="O2", native_arch=False
    )

    if instance_ordinals is None:
        ordinals = tuple(range(len(backend.instances)))
    else:
        ordinals = tuple(instance_ordinals)
    if not ordinals:
        raise NativeFamilyError("at least one family instance must be selected for proof")

    instance_proofs: list[MixedNativeFamilyInstanceProof] = []
    byid = compiler.sequential.node_by_id
    for ordinal in ordinals:
        try:
            inst = backend.instances[ordinal]
            expansion = expand_native_family_instance(compiler, backend, ordinal)
        except (IndexError, NativePlanError) as exc:
            raise NativeFamilyError(str(exc)) from exc
        family_node_ids = {nid for _role, nid in expansion.operation_sequence}
        dslots, islots, bslots, oslots = _initial_hybrid_pools(compiler, typed, family_node_ids)
        callbacks = [None] * len(formula_ids)
        for role, fid in enumerate(formula_ids):
            if not kernel_native[fid] and expansion.role_node_ids[role]:
                callbacks[role] = _python_boundary_callbacks_for_instance_role(
                    compiler, typed, expansion.role_node_ids[role],
                    dslots, islots, bslots, oslots
                )
        bind_payload = _encode_runtime_bindings(inst, sites)
        try:
            reference_payload = encode_reference_instance(reference_plan, ordinal).as_call_args()
        except NativeReferenceError as exc:
            raise NativeFamilyError(str(exc)) from exc
        out_payload = _instance_output_descriptors(typed, backend, expansion)
        variant_arr = np.asarray(inst.variant_ids, dtype=np.int64)
        raw_actual = mod.run(
            dslots,
            islots,
            bslots,
            oslots,
            variant_arr,
            *bind_payload.as_call_args(),
            *reference_payload,
            *out_payload,
            callbacks,
            expansion.operation_count,
        )
        actual = list(raw_actual)
        expected = [
            _authoritative_value(byid[nid]) for _role, nid in expansion.operation_sequence
        ]
        if len(actual) != len(expected):
            raise NativeFamilyError("shared native family output length mismatch")
        exact = all(_hybrid_value_equal(a, e) for a, e in zip(actual, expected))
        max_abs = _hybrid_numeric_error(actual, expected)
        pyops = sum(
            1
            for role, _nid in expansion.operation_sequence
            if not kernel_native[formula_ids[role]]
        )
        instance_proofs.append(
            MixedNativeFamilyInstanceProof(
                instance_ordinal=ordinal,
                block_index=inst.block_index,
                operation_count=expansion.operation_count,
                python_callback_operations=pyops,
                max_abs_error=max_abs,
                exact=exact,
            )
        )

    return MixedNativeFamilyKernelProof(
        family_id=family_id,
        code_signature=backend.kernel.code_signature,
        formula_op_ids=formula_ids,
        variant_count=backend.kernel.variant_count,
        instance_count=len(backend.instances),
        native_formula_ops=sum(kernel_native[f] for f in set(formula_ids)),
        python_formula_ops=sum(not kernel_native[f] for f in set(formula_ids)),
        pyx_path=str(pyx),
        so_path=str(so_path),
        source_lines=source.count("\n"),
        source_bytes=len(source.encode("utf-8")),
        instances=tuple(instance_proofs),
    )
