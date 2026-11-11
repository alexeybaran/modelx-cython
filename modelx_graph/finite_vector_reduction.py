from __future__ import annotations

"""Graph-free scalarization of bounded materialized vector formulas.

This bridge exists for a narrow family that cannot be represented by the older
``StaticSequenceBridge``: a vector is *computed by native recursive formulas*, so
preparing the vector values with Python/modelx would precompute the very state the
compiler is meant to execute.

The bridge instead projects a proven finite vector formula into scalar element
helpers and rewrites consumers onto those helpers.  Native execution then sees only
ordinary scalar Cells plus ``ReductionSpec``-compatible ``range`` generators.  No
Python list, tuple, ``zip`` or filter iterator survives the rewrite.

Accepted producer shapes are deliberately bounded:

* a zero/parameter-arity formula returning one vector or a tuple of vectors built
  from list comprehensions over unit-step ``range`` domains and aligned ``zip``;
* a recursive append builder with finite list-valued base returns, one singleton
  head element, and one unit-step append loop.

The second form is what modelx models often use as a performance representation of
an underlying two-dimensional state.  Scalarizing it restores that semantic state
without introducing a new execution scheduler or vector runtime.
"""

import ast
import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class FiniteVectorReductionError(RuntimeError):
    pass


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


class _SubstituteNames(ast.NodeTransformer):
    def __init__(self, mapping: Mapping[str, ast.AST]):
        self.mapping = dict(mapping)

    def visit_Name(self, node: ast.Name):  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return ast.copy_location(copy.deepcopy(self.mapping[node.id]), node)
        return node


def _substitute(node: ast.AST, mapping: Mapping[str, ast.AST]) -> ast.AST:
    return ast.fix_missing_locations(_SubstituteNames(mapping).visit(copy.deepcopy(node)))


def _strip_docstring(body: Sequence[ast.stmt]) -> list[ast.stmt]:
    out = list(copy.deepcopy(body))
    if (
        out
        and isinstance(out[0], ast.Expr)
        and isinstance(out[0].value, ast.Constant)
        and isinstance(out[0].value.value, str)
    ):
        out = out[1:]
    return out


def _safe_ident(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    if not out or out[0].isdigit():
        out = "v_" + out
    return out


def _helper_uid(source_uid: str, component: int | None) -> str:
    token = f"{source_uid}::finite_vector::{component!r}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    suffix = "vec" if component is None else f"c{component}"
    return f"fvh_{digest}_{_safe_ident(source_uid[-24:])}_{suffix}"


def _const_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -int(node.operand.value)
    if isinstance(node, ast.BinOp):
        left = _const_int(node.left)
        right = _const_int(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    return None


def _add(left: ast.AST, right: ast.AST) -> ast.AST:
    li, ri = _const_int(left), _const_int(right)
    if li == 0:
        return copy.deepcopy(right)
    if ri == 0:
        return copy.deepcopy(left)
    if li is not None and ri is not None:
        return ast.Constant(li + ri)
    # Canonicalize c + (x - c) and (x - c) + c.  Finite builders commonly use
    # one head element plus ``range(1, n)``; keeping this as ``n`` makes exact
    # zip-alignment comparison structural rather than algebraic guesswork.
    if li is not None and isinstance(right, ast.BinOp) and isinstance(right.op, ast.Sub):
        if _const_int(right.right) == li:
            return copy.deepcopy(right.left)
    if ri is not None and isinstance(left, ast.BinOp) and isinstance(left.op, ast.Sub):
        if _const_int(left.right) == ri:
            return copy.deepcopy(left.left)
    return ast.BinOp(copy.deepcopy(left), ast.Add(), copy.deepcopy(right))


def _sub(left: ast.AST, right: ast.AST) -> ast.AST:
    li, ri = _const_int(left), _const_int(right)
    if ri == 0:
        return copy.deepcopy(left)
    if li is not None and ri is not None:
        return ast.Constant(li - ri)
    if ri is not None and isinstance(left, ast.BinOp) and isinstance(left.op, ast.Add):
        if _const_int(left.right) == ri:
            return copy.deepcopy(left.left)
        if _const_int(left.left) == ri:
            return copy.deepcopy(left.right)
    return ast.BinOp(copy.deepcopy(left), ast.Sub(), copy.deepcopy(right))


def _range_geometry(
    call: ast.Call,
    *,
    scalar_env: Mapping[str, ast.AST],
) -> tuple[ast.AST, ast.AST] | None:
    """Return ``(index_expr(i), length_expr)`` for unit-step range."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "range" and not call.keywords):
        return None
    args = [ast.fix_missing_locations(_substitute(arg, scalar_env)) for arg in call.args]
    if len(args) == 1:
        start, stop, step = ast.Constant(0), args[0], 1
    elif len(args) == 2:
        start, stop, step = args[0], args[1], 1
    elif len(args) == 3:
        step = _const_int(args[2])
        start, stop = args[0], args[1]
    else:
        return None
    if step != 1:
        return None
    index = _add(start, ast.Name("__fv_i", ast.Load()))
    length = _sub(stop, start)
    return ast.fix_missing_locations(index), ast.fix_missing_locations(length)


@dataclass(frozen=True)
class FiniteVectorElementHelper:
    source_uid: str
    source_name: str
    component: int | None
    uid: str
    source_parameters: tuple[str, ...]
    element_parameter: str
    length_expr: ast.AST
    function: ast.FunctionDef
    provenance: str
    dtype: str = "object"
    max_length: int | None = None
    parameter_dtypes: tuple[str, ...] = ()
    parameter_ranges: tuple[tuple[int, int] | None, ...] = ()

    @property
    def parameters(self) -> tuple[str, ...]:
        return (*self.source_parameters, self.element_parameter)

    @property
    def key(self) -> tuple[str, int | None]:
        return self.source_uid, self.component

    def manifest(self) -> dict[str, Any]:
        return {
            "source_uid": self.source_uid,
            "source_name": self.source_name,
            "component": self.component,
            "uid": self.uid,
            "source_parameters": list(self.source_parameters),
            "element_parameter": self.element_parameter,
            "length_expr": ast.unparse(self.length_expr),
            "function": ast.unparse(self.function),
            "provenance": self.provenance,
            "dtype": self.dtype,
            "max_length": self.max_length,
            "parameter_dtypes": list(self.parameter_dtypes),
            "parameter_ranges": [None if row is None else list(row) for row in self.parameter_ranges],
            "runtime_vector_object": False,
        }


@dataclass(frozen=True)
class _VectorExpr:
    length_expr: ast.AST
    element_expr: ast.AST


def _listcomp_vector(
    node: ast.AST,
    *,
    range_env: Mapping[str, ast.Call],
    vector_env: Mapping[str, _VectorExpr],
    scalar_env: Mapping[str, ast.AST],
) -> _VectorExpr | None:
    if not isinstance(node, ast.ListComp) or len(node.generators) != 1:
        return None
    comp = node.generators[0]
    if comp.ifs or comp.is_async:
        return None
    iterator = comp.iter
    if isinstance(iterator, ast.Name) and iterator.id in range_env:
        iterator = range_env[iterator.id]
    if isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name) and iterator.func.id == "range":
        if not isinstance(comp.target, ast.Name):
            return None
        geometry = _range_geometry(iterator, scalar_env=scalar_env)
        if geometry is None:
            return None
        index_expr, length_expr = geometry
        elt = _substitute(node.elt, {comp.target.id: index_expr})
        return _VectorExpr(length_expr, elt)
    if (
        isinstance(iterator, ast.Call)
        and isinstance(iterator.func, ast.Name)
        and iterator.func.id == "zip"
        and not iterator.keywords
        and isinstance(comp.target, (ast.Tuple, ast.List))
        and len(comp.target.elts) == len(iterator.args)
    ):
        sources: list[_VectorExpr] = []
        for arg in iterator.args:
            if not isinstance(arg, ast.Name) or arg.id not in vector_env:
                return None
            sources.append(vector_env[arg.id])
        if not sources:
            return None
        if any(_dump(source.length_expr) != _dump(sources[0].length_expr) for source in sources[1:]):
            return None
        mapping: dict[str, ast.AST] = {}
        for target, source in zip(comp.target.elts, sources):
            if not isinstance(target, ast.Name):
                return None
            mapping[target.id] = source.element_expr
        return _VectorExpr(copy.deepcopy(sources[0].length_expr), _substitute(node.elt, mapping))
    return None


def build_comprehension_vector_helpers(
    *,
    source_uid: str,
    source_name: str,
    parameters: Sequence[str],
    function: ast.FunctionDef,
) -> tuple[FiniteVectorElementHelper, ...]:
    """Recognize tuple/direct vectors composed from finite list comprehensions."""
    body = _strip_docstring(function.body)
    if not body or not isinstance(body[-1], ast.Return):
        return ()
    range_env: dict[str, ast.Call] = {}
    vector_env: dict[str, _VectorExpr] = {}
    scalar_env: dict[str, ast.AST] = {}
    for st in body[:-1]:
        if not (
            isinstance(st, ast.Assign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], ast.Name)
        ):
            return ()
        name = st.targets[0].id
        value = st.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "range":
            if _range_geometry(value, scalar_env=scalar_env) is None:
                return ()
            range_env[name] = copy.deepcopy(value)
            continue
        vector = _listcomp_vector(
            value, range_env=range_env, vector_env=vector_env, scalar_env=scalar_env
        )
        if vector is not None:
            vector_env[name] = vector
            continue
        # Allow primitive scalar aliases only; vector builders must not hide an
        # arbitrary provider/iterator behind local object state.
        if any(isinstance(n, (ast.ListComp, ast.GeneratorExp, ast.For)) for n in ast.walk(value)):
            return ()
        scalar_env[name] = ast.fix_missing_locations(_substitute(value, scalar_env))

    ret = body[-1].value
    components: list[tuple[int | None, _VectorExpr]] = []
    values = list(ret.elts) if isinstance(ret, (ast.Tuple, ast.List)) else [ret]
    tuple_result = isinstance(ret, ast.Tuple)
    for pos, value in enumerate(values):
        vector: _VectorExpr | None = None
        if isinstance(value, ast.Name):
            vector = vector_env.get(value.id)
        if vector is None:
            vector = _listcomp_vector(
                value, range_env=range_env, vector_env=vector_env, scalar_env=scalar_env
            )
        if vector is None:
            return ()
        components.append((pos if tuple_result else None, vector))

    out: list[FiniteVectorElementHelper] = []
    for component, vector in components:
        uid = _helper_uid(source_uid, component)
        fn = ast.FunctionDef(
            name=uid,
            args=ast.arguments(
                posonlyargs=[],
                args=[*(ast.arg(arg=name) for name in parameters), ast.arg(arg="__fv_i")],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=[ast.Return(copy.deepcopy(vector.element_expr))],
            decorator_list=[],
        )
        out.append(FiniteVectorElementHelper(
            source_uid=source_uid,
            source_name=source_name,
            component=component,
            uid=uid,
            source_parameters=tuple(parameters),
            element_parameter="__fv_i",
            length_expr=ast.fix_missing_locations(copy.deepcopy(vector.length_expr)),
            function=ast.fix_missing_locations(fn),
            provenance="finite_listcomp_or_aligned_zip",
        ))
    return tuple(out)


@dataclass(frozen=True)
class _BoundVector:
    helper: FiniteVectorElementHelper
    args: tuple[ast.AST, ...]

    def length(self) -> ast.AST:
        mapping = {
            name: arg for name, arg in zip(self.helper.source_parameters, self.args)
        }
        return _substitute(self.helper.length_expr, mapping)

    def element(self, index: ast.AST) -> ast.Call:
        return ast.Call(
            ast.Name(self.helper.uid, ast.Load()),
            [*(copy.deepcopy(arg) for arg in self.args), copy.deepcopy(index)],
            [],
        )


def _parse_provider_expr(
    node: ast.AST,
    helpers: Mapping[tuple[str, int | None], FiniteVectorElementHelper],
) -> _BoundVector | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        helper = helpers.get((node.func.id, None))
        if helper is not None and len(node.args) == len(helper.source_parameters):
            return _BoundVector(helper, tuple(copy.deepcopy(node.args)))
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and not node.value.keywords
    ):
        comp = _const_int(node.slice)
        helper = helpers.get((node.value.func.id, comp)) if comp is not None else None
        if helper is not None and len(node.value.args) == len(helper.source_parameters):
            return _BoundVector(helper, tuple(copy.deepcopy(node.value.args)))
    return None


class _VectorAliasExprRewriter(ast.NodeTransformer):
    def __init__(self, aliases: Mapping[str, _BoundVector]):
        self.aliases = dict(aliases)

    def visit_Subscript(self, node: ast.Subscript):  # noqa: N802
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in self.aliases:
            return ast.copy_location(self.aliases[node.value.id].element(node.slice), node)
        return node

    def visit_Call(self, node: ast.Call):  # noqa: N802
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.aliases
        ):
            return ast.copy_location(self.aliases[node.args[0].id].length(), node)
        return self.generic_visit(node)


def _rewrite_alias_expr(node: ast.AST, aliases: Mapping[str, _BoundVector]) -> ast.AST:
    return ast.fix_missing_locations(_VectorAliasExprRewriter(aliases).visit(copy.deepcopy(node)))


def _scalarize_list_return(
    value: ast.AST,
    *,
    scalar_env: Mapping[str, ast.AST],
    range_env: Mapping[str, ast.Call],
    index_name: str,
    expected_length: ast.AST,
) -> ast.AST | None:
    if (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Mult)
        and isinstance(value.left, (ast.List, ast.Tuple))
        and len(value.left.elts) == 1
    ):
        expected_len = ast.fix_missing_locations(_substitute(value.right, scalar_env))
        if _dump(expected_len) != _dump(expected_length):
            return None
        return copy.deepcopy(value.left.elts[0])
    if isinstance(value, ast.ListComp):
        vector = _listcomp_vector(
            value, range_env=range_env, vector_env={}, scalar_env=scalar_env
        )
        if vector is None:
            return None
        if _dump(vector.length_expr) != _dump(expected_length):
            return None
        return _substitute(vector.element_expr, {"__fv_i": ast.Name(index_name, ast.Load())})
    return None


def build_recursive_append_vector_helper(
    *,
    source_uid: str,
    source_name: str,
    parameters: Sequence[str],
    function: ast.FunctionDef,
    known_helpers: Mapping[tuple[str, int | None], FiniteVectorElementHelper],
) -> FiniteVectorElementHelper | None:
    """Project one bounded singleton-head + append vector recurrence to an element Cell."""
    body = _strip_docstring(function.body)
    if len(body) < 4 or not isinstance(body[-1], ast.Return) or not isinstance(body[-1].value, ast.Name):
        return None
    out_name = body[-1].value.id
    out_pos = None
    for i, st in enumerate(body[:-1]):
        if (
            isinstance(st, ast.Assign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], ast.Name)
            and st.targets[0].id == out_name
            and isinstance(st.value, ast.List)
            and len(st.value.elts) == 1
        ):
            out_pos = i
    if out_pos is None or out_pos + 2 != len(body) - 1:
        return None
    loop = body[out_pos + 1]
    if not (
        isinstance(loop, ast.For)
        and isinstance(loop.target, ast.Name)
        and len(loop.body) == 1
        and isinstance(loop.body[0], ast.Expr)
        and isinstance(loop.body[0].value, ast.Call)
        and isinstance(loop.body[0].value.func, ast.Attribute)
        and isinstance(loop.body[0].value.func.value, ast.Name)
        and loop.body[0].value.func.value.id == out_name
        and loop.body[0].value.func.attr == "append"
        and len(loop.body[0].value.args) == 1
        and not loop.body[0].value.keywords
        and isinstance(loop.iter, ast.Call)
    ):
        return None

    scalar_env: dict[str, ast.AST] = {}
    range_env: dict[str, ast.Call] = {}
    aliases: dict[str, _BoundVector] = {}
    prefix: list[ast.stmt] = []
    # First gather simple top-level aliases before the output builder. If/return
    # branches are retained and scalarized after the common vector length is known.
    for st in body[:out_pos]:
        if isinstance(st, ast.If):
            prefix.append(copy.deepcopy(st))
            continue
        if not (
            isinstance(st, ast.Assign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], ast.Name)
        ):
            return None
        name = st.targets[0].id
        bound = _parse_provider_expr(st.value, known_helpers)
        if bound is None and (
            isinstance(st.value, ast.Call)
            and isinstance(st.value.func, ast.Name)
            and st.value.func.id == source_uid
            and not st.value.keywords
        ):
            # Self binding becomes available after the common length is known;
            # retain a marker for now.
            aliases[name] = ("__SELF__", tuple(copy.deepcopy(st.value.args)))  # type: ignore[assignment]
            continue
        if bound is not None:
            aliases[name] = bound
            continue
        if isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Name) and st.value.func.id == "range":
            range_env[name] = copy.deepcopy(st.value)
        else:
            scalar_env[name] = ast.fix_missing_locations(_substitute(st.value, scalar_env))
            prefix.append(copy.deepcopy(st))

    geometry = _range_geometry(loop.iter, scalar_env=scalar_env)
    if geometry is None:
        return None
    loop_index, loop_count = geometry
    start_expr = _substitute(loop_index, {"__fv_i": ast.Constant(0)})
    if _const_int(start_expr) != 1:
        return None
    # One head element plus range(1, stop) append iterations => stop elements.
    length_expr = _add(ast.Constant(1), loop_count)
    length_expr = ast.fix_missing_locations(length_expr)
    helper_uid = _helper_uid(source_uid, None)
    self_helper = FiniteVectorElementHelper(
        source_uid=source_uid,
        source_name=source_name,
        component=None,
        uid=helper_uid,
        source_parameters=tuple(parameters),
        element_parameter="__fv_i",
        length_expr=copy.deepcopy(length_expr),
        function=ast.FunctionDef(name=helper_uid,args=ast.arguments(posonlyargs=[],args=[],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=[],decorator_list=[]),
        provenance="bounded_recursive_singleton_append",
    )
    normalized_aliases: dict[str, _BoundVector] = {}
    for name, value in aliases.items():
        if isinstance(value, tuple) and value and value[0] == "__SELF__":
            normalized_aliases[name] = _BoundVector(self_helper, value[1])  # type: ignore[index]
        else:
            normalized_aliases[name] = value  # type: ignore[assignment]

    # Every external vector alias must have the same structural length. This is
    # the exact alignment proof used later for zip semantics.
    for bound in normalized_aliases.values():
        if bound.helper.source_uid == source_uid:
            continue
        if _dump(bound.length()) != _dump(length_expr):
            return None

    index_name = "__fv_i"

    def transform_branch(st: ast.stmt) -> ast.stmt | None:
        if isinstance(st, ast.Return):
            scalar = _scalarize_list_return(
                st.value, scalar_env=scalar_env, range_env=range_env,
                index_name=index_name, expected_length=length_expr,
            )
            if scalar is None:
                return None
            return ast.Return(_rewrite_alias_expr(scalar, normalized_aliases))
        if isinstance(st, ast.Assign):
            return copy.deepcopy(st)
        if isinstance(st, ast.If):
            body2: list[ast.stmt] = []
            for child in st.body:
                got = transform_branch(child)
                if got is None:
                    return None
                body2.append(got)
            else2: list[ast.stmt] = []
            for child in st.orelse:
                got = transform_branch(child)
                if got is None:
                    return None
                else2.append(got)
            return ast.If(copy.deepcopy(st.test), body2, else2)
        return None

    transformed_prefix: list[ast.stmt] = []
    for st in prefix:
        got = transform_branch(st)
        if got is None:
            return None
        transformed_prefix.append(got)

    append_expr = copy.deepcopy(loop.body[0].value.args[0])
    append_expr = _substitute(append_expr, {loop.target.id: ast.Name(index_name, ast.Load())})
    append_expr = _rewrite_alias_expr(append_expr, normalized_aliases)
    head_expr = _rewrite_alias_expr(body[out_pos].value.elts[0], normalized_aliases)  # type: ignore[union-attr]
    final = ast.Return(ast.IfExp(
        ast.Compare(ast.Name(index_name, ast.Load()), [ast.Eq()], [ast.Constant(0)]),
        head_expr,
        append_expr,
    ))
    fn = ast.FunctionDef(
        name=helper_uid,
        args=ast.arguments(
            posonlyargs=[],
            args=[*(ast.arg(arg=name) for name in parameters), ast.arg(arg=index_name)],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[*transformed_prefix, final],
        decorator_list=[],
    )
    return FiniteVectorElementHelper(
        source_uid=source_uid,
        source_name=source_name,
        component=None,
        uid=helper_uid,
        source_parameters=tuple(parameters),
        element_parameter=index_name,
        length_expr=length_expr,
        function=ast.fix_missing_locations(fn),
        provenance="bounded_recursive_singleton_append",
    )


class FiniteVectorConsumerNormalizer(ast.NodeTransformer):
    """Rewrite vector aliases, element access and finite zip/sum to scalar helpers."""

    def __init__(self, helpers: Mapping[tuple[str, int | None], FiniteVectorElementHelper]):
        self.helpers = dict(helpers)

    def _bound(self, node: ast.AST) -> _BoundVector | None:
        return _parse_provider_expr(node, self.helpers)

    @staticmethod
    def _fresh_index(fn: ast.FunctionDef) -> str:
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        base = "__fv_i"
        if base not in names:
            return base
        i = 2
        while f"{base}{i}" in names:
            i += 1
        return f"{base}{i}"

    def normalize_function(self, fn: ast.FunctionDef) -> ast.FunctionDef:
        body = list(copy.deepcopy(fn.body))
        doc: list[ast.stmt] = []
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            doc = [body.pop(0)]
        aliases: dict[str, _BoundVector] = {}
        kept: list[ast.stmt] = []
        for st in body:
            if isinstance(st, ast.Return):
                kept.append(ast.Return(self._rewrite_return(st.value, aliases, fn)))
                continue
            if isinstance(st, ast.Assign) and len(st.targets) == 1:
                target = st.targets[0]
                if isinstance(target, ast.Name):
                    bound = self._bound(st.value)
                    if bound is not None:
                        aliases[target.id] = bound
                        continue
                if isinstance(target, (ast.Tuple, ast.List)) and isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Name):
                    component_bounds: list[_BoundVector] = []
                    for i, element in enumerate(target.elts):
                        helper = self.helpers.get((st.value.func.id, i))
                        if not isinstance(element, ast.Name) or helper is None or len(st.value.args) != len(helper.source_parameters):
                            component_bounds = []
                            break
                        component_bounds.append(_BoundVector(helper, tuple(copy.deepcopy(st.value.args))))
                    if component_bounds:
                        for element, bound in zip(target.elts, component_bounds):
                            assert isinstance(element, ast.Name)
                            aliases[element.id] = bound
                        continue
            kept.append(_VectorAliasExprRewriter(aliases).visit(copy.deepcopy(st)))
        out = copy.deepcopy(fn)
        out.body = [*doc, *kept]
        return ast.fix_missing_locations(out)

    def _rewrite_return(
        self, value: ast.AST, aliases: Mapping[str, _BoundVector], fn: ast.FunctionDef
    ) -> ast.AST:
        # sum(vector_provider(...)) or sum(local_vector_alias)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "sum" and len(value.args) == 1 and not value.keywords:
            direct = self._bound(value.args[0])
            if direct is None and isinstance(value.args[0], ast.Name):
                direct = aliases.get(value.args[0].id)
            if direct is not None:
                idx = self._fresh_index(fn)
                gen = ast.GeneratorExp(
                    direct.element(ast.Name(idx, ast.Load())),
                    [ast.comprehension(
                        target=ast.Name(idx, ast.Store()),
                        iter=ast.Call(ast.Name("range", ast.Load()), [direct.length()], []),
                        ifs=[], is_async=0,
                    )],
                )
                return ast.copy_location(ast.Call(ast.Name("sum", ast.Load()), [gen], []), value)

            gen = value.args[0]
            if isinstance(gen, ast.GeneratorExp) and len(gen.generators) == 1:
                comp = gen.generators[0]
                if (
                    isinstance(comp.iter, ast.Call)
                    and isinstance(comp.iter.func, ast.Name)
                    and comp.iter.func.id == "zip"
                    and not comp.iter.keywords
                    and isinstance(comp.target, (ast.Tuple, ast.List))
                    and len(comp.target.elts) == len(comp.iter.args)
                ):
                    bounds: list[_BoundVector] = []
                    for arg in comp.iter.args:
                        bound = aliases.get(arg.id) if isinstance(arg, ast.Name) else self._bound(arg)
                        if bound is None:
                            bounds = []
                            break
                        bounds.append(bound)
                    if bounds and all(_dump(b.length()) == _dump(bounds[0].length()) for b in bounds[1:]):
                        idx = self._fresh_index(fn)
                        mapping: dict[str, ast.AST] = {}
                        for target, bound in zip(comp.target.elts, bounds):
                            if not isinstance(target, ast.Name):
                                return _rewrite_alias_expr(value, aliases)
                            mapping[target.id] = bound.element(ast.Name(idx, ast.Load()))
                        elt = _substitute(gen.elt, mapping)
                        ifs = [_substitute(test, mapping) for test in comp.ifs]
                        new_gen = ast.GeneratorExp(
                            elt,
                            [ast.comprehension(
                                target=ast.Name(idx, ast.Store()),
                                iter=ast.Call(ast.Name("range", ast.Load()), [bounds[0].length()], []),
                                ifs=ifs,
                                is_async=0,
                            )],
                        )
                        return ast.copy_location(ast.Call(ast.Name("sum", ast.Load()), [new_gen], []), value)
        return _rewrite_alias_expr(value, aliases)



class _ReplaceMinOfName(ast.NodeTransformer):
    def __init__(self, source_name: str, replacement_name: str):
        self.source_name = source_name
        self.replacement_name = replacement_name
        self.replacements = 0

    def visit_Call(self, node: ast.Call):  # noqa: N802
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "min"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == self.source_name
        ):
            self.replacements += 1
            return ast.copy_location(ast.Name(self.replacement_name, ast.Load()), node)
        return self.generic_visit(node)


def _fresh_local_name(fn: ast.FunctionDef, stem: str) -> str:
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    if stem not in used:
        return stem
    i = 2
    while f"{stem}_{i}" in used:
        i += 1
    return f"{stem}_{i}"


def _pure_scalar_filter_expr(node: ast.AST, item_name: str) -> bool:
    """Narrow side-effect-free filter accepted by the streaming-min bridge."""

    allowed = (
        ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare,
        ast.Name, ast.Load, ast.Constant, ast.And, ast.Or, ast.Not,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
        ast.UAdd, ast.USub, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    )
    for child in ast.walk(node):
        if not isinstance(child, allowed):
            return False
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id != item_name:
            return False
    return True


def normalize_bounded_filtered_min_listcomps(fn: ast.FunctionDef) -> ast.FunctionDef:
    """Scalarize one bounded materialized-list filtered minimum in a Cell body.

    Accepted source block::

        xs = [VALUE(i) for i in range(...)]
        xs = [x for x in xs if PRED(x)]
        if not xs:
            return FALLBACK
        ... min(xs) ...

    ``VALUE`` is restricted to one direct scalar Cell/helper call and ``PRED`` to
    side-effect-free scalar syntax over the filtered item.  The bridge streams the
    values into a native minimum accumulator, preserving the explicit empty path
    and eliminating Python list materialization.  Any other use of ``xs`` leaves
    the function unchanged.
    """

    body = list(copy.deepcopy(fn.body))
    for pos in range(max(0, len(body) - 2)):
        first, second, empty = body[pos:pos + 3]
        if not (
            isinstance(first, ast.Assign)
            and len(first.targets) == 1
            and isinstance(first.targets[0], ast.Name)
            and isinstance(first.value, ast.ListComp)
            and len(first.value.generators) == 1
            and isinstance(second, ast.Assign)
            and len(second.targets) == 1
            and isinstance(second.targets[0], ast.Name)
            and second.targets[0].id == first.targets[0].id
            and isinstance(second.value, ast.ListComp)
            and len(second.value.generators) == 1
            and isinstance(empty, ast.If)
            and not empty.orelse
            and len(empty.body) == 1
            and isinstance(empty.body[0], ast.Return)
        ):
            continue
        list_name = first.targets[0].id
        c1 = first.value.generators[0]
        c2 = second.value.generators[0]
        if not (
            isinstance(c1.target, ast.Name)
            and isinstance(c1.iter, ast.Call)
            and isinstance(c1.iter.func, ast.Name)
            and c1.iter.func.id == "range"
            and not c1.iter.keywords
            and 1 <= len(c1.iter.args) <= 3
            and not c1.ifs
            and not c1.is_async
            and isinstance(first.value.elt, ast.Call)
            and isinstance(first.value.elt.func, ast.Name)
            and not first.value.elt.keywords
            and isinstance(c2.target, ast.Name)
            and isinstance(c2.iter, ast.Name)
            and c2.iter.id == list_name
            and not c2.is_async
            and isinstance(second.value.elt, ast.Name)
            and second.value.elt.id == c2.target.id
            and len(c2.ifs) == 1
            and _pure_scalar_filter_expr(c2.ifs[0], c2.target.id)
            and isinstance(empty.test, ast.UnaryOp)
            and isinstance(empty.test.op, ast.Not)
            and isinstance(empty.test.operand, ast.Name)
            and empty.test.operand.id == list_name
        ):
            continue

        rest = body[pos + 3:]
        best_name = _fresh_local_name(fn, f"__fmin_best_{list_name}")
        rewriter = _ReplaceMinOfName(list_name, best_name)
        rewritten_rest = [ast.fix_missing_locations(rewriter.visit(copy.deepcopy(st))) for st in rest]
        if rewriter.replacements == 0:
            continue
        if any(
            isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == list_name
            for st in rewritten_rest for n in ast.walk(st)
        ):
            continue

        found_name = _fresh_local_name(fn, f"__fmin_found_{list_name}")
        item_name = _fresh_local_name(fn, f"__fmin_item_{list_name}")
        # Substitute the second-comprehension binder with the generated scalar item.
        predicate = _substitute(c2.ifs[0], {c2.target.id: ast.Name(item_name, ast.Load())})
        loop = ast.For(
            target=copy.deepcopy(c1.target),
            iter=copy.deepcopy(c1.iter),
            body=[
                ast.Assign([ast.Name(item_name, ast.Store())], copy.deepcopy(first.value.elt)),
                ast.If(
                    test=predicate,
                    body=[ast.If(
                        test=ast.BoolOp(ast.Or(), [
                            ast.UnaryOp(ast.Not(), ast.Name(found_name, ast.Load())),
                            ast.Compare(
                                ast.Name(item_name, ast.Load()), [ast.Lt()],
                                [ast.Name(best_name, ast.Load())],
                            ),
                        ]),
                        body=[
                            ast.Assign([ast.Name(best_name, ast.Store())], ast.Name(item_name, ast.Load())),
                            ast.Assign([ast.Name(found_name, ast.Store())], ast.Constant(True)),
                        ],
                        orelse=[],
                    )],
                    orelse=[],
                ),
            ],
            orelse=[],
        )
        replacement = [
            ast.Assign([ast.Name(found_name, ast.Store())], ast.Constant(False)),
            ast.Assign([ast.Name(best_name, ast.Store())], ast.Constant(0.0)),
            loop,
            ast.If(
                ast.UnaryOp(ast.Not(), ast.Name(found_name, ast.Load())),
                copy.deepcopy(empty.body), [],
            ),
            *rewritten_rest,
        ]
        out = copy.deepcopy(fn)
        out.body = [*body[:pos], *replacement]
        return ast.fix_missing_locations(out)
    return ast.fix_missing_locations(copy.deepcopy(fn))

def normalize_finite_vector_consumers(
    fn: ast.FunctionDef,
    helpers: Mapping[tuple[str, int | None], FiniteVectorElementHelper],
) -> ast.FunctionDef:
    return FiniteVectorConsumerNormalizer(helpers).normalize_function(fn)


__all__ = [
    "FiniteVectorReductionError",
    "FiniteVectorElementHelper",
    "build_comprehension_vector_helpers",
    "build_recursive_append_vector_helper",
    "FiniteVectorConsumerNormalizer",
    "normalize_finite_vector_consumers",
    "normalize_bounded_filtered_min_listcomps",
]
