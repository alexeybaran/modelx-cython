from __future__ import annotations

"""Graph-free lowering for frozen scalar sequence-provider access.

The bridge accepts only zero-argument formula providers, literal Python
index/slice syntax and an optional primitive numeric cast. It evaluates the
complete RunDomain during preparation and registers a typed point buffer; no
Python sequence enters native recursive execution.
"""

import ast
import copy
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .semantic_reuse import UnrollLiteralReductions


class StaticSequenceBridgeError(RuntimeError):
    pass


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, (bool, int, str))
    ):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
            return -int(node.operand.value)
    raise StaticSequenceBridgeError("static sequence access requires literal indices")


def _selector(node: ast.AST) -> int | slice:
    if isinstance(node, ast.Slice):
        lower = _literal(node.lower)
        upper = _literal(node.upper)
        step = _literal(node.step)
        if any(x is not None and not isinstance(x, int) for x in (lower, upper, step)):
            raise StaticSequenceBridgeError("static sequence slice bounds must be integers")
        if step == 0:
            raise StaticSequenceBridgeError("static sequence slice step cannot be zero")
        return slice(lower, upper, step)
    value = _literal(node)
    if not isinstance(value, int):
        raise StaticSequenceBridgeError("static sequence index must be an integer")
    return value


def _numeric(value: Any, cast_name: str | None) -> tuple[str, Any]:
    if cast_name is not None:
        value = {"int": int, "float": float, "bool": bool}[cast_name](value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return "bool", bool(value)
    if isinstance(value, (int, np.integer)):
        return "int64", int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not np.isfinite(value):
            raise StaticSequenceBridgeError("static sequence access produced a non-finite scalar")
        return "float64", value
    raise StaticSequenceBridgeError(
        f"static sequence access produced unsupported {type(value).__name__}; "
        "use an explicit primitive numeric cast"
    )


class FrozenSequenceAccessNormalizer(ast.NodeTransformer):
    """Replace proven sequence-provider scalar access by typed point inputs."""

    def __init__(
        self,
        *,
        uid_to_source_name: Mapping[str, str],
        run_keys: Sequence[Any],
        space_instance: Callable[[Any], Any],
        register_point_input: Callable[[tuple[Any, ...], str, Callable[[Sequence[Any]], Any], str], str],
    ) -> None:
        self.uid_to_source_name = dict(uid_to_source_name)
        self.run_keys = tuple(run_keys)
        self.space_instance = space_instance
        self.register_point_input = register_point_input

    @staticmethod
    def _shape(node: ast.AST) -> tuple[ast.Subscript, str | None] | None:
        if isinstance(node, ast.Subscript):
            return node, None
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"int", "float", "bool"}
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Subscript)
        ):
            return node.args[0], node.func.id
        return None

    def generic_visit(self, node: ast.AST):
        shaped = self._shape(node)
        if shaped is not None:
            subscript, cast_name = shaped
            provider = subscript.value
            if (
                isinstance(provider, ast.Call)
                and isinstance(provider.func, ast.Name)
                and provider.func.id in self.uid_to_source_name
                and not provider.args
                and not provider.keywords
            ):
                try:
                    selector = _selector(subscript.slice)
                except StaticSequenceBridgeError:
                    # This is not the bridge's literal sequence subset (for
                    # example ``model_point()['age']``); leave it for the mature
                    # point/table normalizers instead of poisoning the formula.
                    return super().generic_visit(node)
                uid = provider.func.id
                source_name = self.uid_to_source_name[uid]

                def evaluate(keys: Sequence[Any]) -> np.ndarray:
                    values: list[Any] = []
                    kinds: list[str] = []
                    for key in keys:
                        source = getattr(self.space_instance(key), source_name)()
                        kind, value = _numeric(source[selector], cast_name)
                        kinds.append(kind)
                        values.append(value)
                    if not values:
                        raise StaticSequenceBridgeError(
                            "static sequence access requires a non-empty RunDomain"
                        )
                    dtype = "float64" if "float64" in kinds else "int64" if "int64" in kinds else "bool"
                    np_dtype = np.float64 if dtype == "float64" else np.int64 if dtype == "int64" else np.bool_
                    return np.asarray(values, dtype=np_dtype)

                sample = evaluate(self.run_keys)
                dtype = "bool" if sample.dtype.kind == "b" else "int64" if sample.dtype.kind in "iu" else "float64"
                token = (
                    "static_sequence_point_scalar",
                    uid,
                    ast.dump(subscript.slice, include_attributes=False),
                    cast_name,
                )
                key = self.register_point_input(
                    token,
                    dtype,
                    evaluate,
                    f"frozen sequence scalar {source_name}[{ast.unparse(subscript.slice)}]",
                )
                return ast.copy_location(
                    ast.Call(ast.Name("point_input", ast.Load()), [ast.Constant(key)], []),
                    node,
                )
        return super().generic_visit(node)


class FrozenSequenceMembershipNormalizer(ast.NodeTransformer):
    """Lower bounded finite sequence membership to prepared point buffers."""

    def __init__(self, *, uid_to_source_name, run_keys, space_instance, register_point_input, max_slots: int = 16):
        self.uid_to_source_name = dict(uid_to_source_name)
        self.run_keys = tuple(run_keys)
        self.space_instance = space_instance
        self.register_point_input = register_point_input
        self.max_slots = int(max_slots)

    def _snapshot(self, uid, keys):
        source_name = self.uid_to_source_name[uid]
        rows=[]
        for key in keys:
            value=getattr(self.space_instance(key), source_name)()
            if not isinstance(value,(tuple,list)):
                return None
            row=tuple(value)
            if len(row)>self.max_slots:
                return None
            for item in row:
                try: _numeric(item,None)
                except StaticSequenceBridgeError: return None
            rows.append(row)
        return tuple(rows)

    def visit_Compare(self,node):  # noqa: N802
        if len(node.ops)!=1 or len(node.comparators)!=1 or not isinstance(node.ops[0],(ast.In,ast.NotIn)):
            return self.generic_visit(node)
        provider=node.comparators[0]
        if not (isinstance(provider,ast.Call) and isinstance(provider.func,ast.Name) and provider.func.id in self.uid_to_source_name and not provider.args and not provider.keywords):
            return self.generic_visit(node)
        uid=provider.func.id; source_name=self.uid_to_source_name[uid]
        snap=self._snapshot(uid,self.run_keys)
        if snap is None: return self.generic_visit(node)
        slots=max((len(r) for r in snap),default=0)
        if not slots: result=ast.Constant(False)
        else:
            kinds=[_numeric(x,None)[0] for r in snap for x in r]
            dtype='float64' if 'float64' in kinds else 'int64' if 'int64' in kinds else 'bool'
            np_dtype=np.float64 if dtype=='float64' else np.int64 if dtype=='int64' else np.bool_
            query=self.visit(node.left); terms=[]
            for slot in range(slots):
                def values_getter(keys,slot=slot,uid=uid,np_dtype=np_dtype):
                    current=self._snapshot(uid,keys)
                    if current is None: raise StaticSequenceBridgeError(f"frozen sequence membership provider {source_name!r} changed shape/type")
                    vals=[]
                    for row in current:
                        item=row[slot] if slot<len(row) else 0
                        _kind,numeric=_numeric(item,None); vals.append(numeric)
                    return np.asarray(vals,dtype=np_dtype)
                def valid_getter(keys,slot=slot,uid=uid):
                    current=self._snapshot(uid,keys)
                    if current is None: raise StaticSequenceBridgeError(f"frozen sequence membership provider {source_name!r} changed shape/type")
                    return np.asarray([slot<len(r) for r in current],dtype=np.bool_)
                value_key=self.register_point_input(("static_sequence_membership_value",uid,slot,dtype),dtype,values_getter,f"frozen sequence membership value {source_name}[slot={slot}]")
                valid_key=self.register_point_input(("static_sequence_membership_valid",uid,slot),"bool",valid_getter,f"frozen sequence membership validity {source_name}[slot={slot}]")
                terms.append(ast.BoolOp(ast.And(),[
                    ast.Call(ast.Name('point_input',ast.Load()),[ast.Constant(valid_key)],[]),
                    ast.Compare(copy.deepcopy(query),[ast.Eq()],[ast.Call(ast.Name('point_input',ast.Load()),[ast.Constant(value_key)],[])])
                ]))
            result=terms[0] if len(terms)==1 else ast.BoolOp(ast.Or(),terms)
        if isinstance(node.ops[0],ast.NotIn): result=ast.UnaryOp(ast.Not(),result)
        return ast.copy_location(ast.fix_missing_locations(result),node)


class FrozenIntegerSequenceReductionNormalizer(ast.NodeTransformer):
    """Rewrite small invariant integer provider iterators to literal tuples.

    This is intentionally narrower than a generic Python iterable lowering. A
    zero-argument Cell provider must return the *same* small finite integer
    sequence on every key in the complete RunDomain. The caller then reuses
    ``semantic_reuse.UnrollLiteralReductions`` so no iterable runtime or new
    reduction IR is introduced.

    Examples admitted by this bridge::

        sum(sa_pp(t, i) for i in subaccount_ids())

    when ``subaccount_ids()`` is proven to be ``[1, 2]`` over the full RunDomain.
    Category/string, varying, or larger iterators are deliberately left untouched
    for richer finite-sequence reduction handling.
    """

    def __init__(
        self,
        *,
        uid_to_source_name: Mapping[str, str],
        run_keys: Sequence[Any],
        space_instance: Callable[[Any], Any],
        max_slots: int = 16,
    ) -> None:
        self.uid_to_source_name = dict(uid_to_source_name)
        self.run_keys = tuple(run_keys)
        self.space_instance = space_instance
        self.max_slots = int(max_slots)

    def _sequence(self, uid: str) -> tuple[int, ...] | None:
        source_name = self.uid_to_source_name[uid]
        baseline: tuple[int, ...] | None = None
        for key in self.run_keys:
            value = getattr(self.space_instance(key), source_name)()
            if not isinstance(value, (tuple, list)):
                return None
            if len(value) > self.max_slots:
                return None
            row: list[int] = []
            for item in value:
                if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                    return None
                row.append(int(item))
            current = tuple(row)
            if baseline is None:
                baseline = current
            elif current != baseline:
                return None
        return baseline

    def visit_GeneratorExp(self, node):  # noqa: N802
        node = self.generic_visit(node)
        if len(node.generators) != 1:
            return node
        comp = node.generators[0]
        provider = comp.iter
        if not (
            isinstance(comp.target, ast.Name)
            and isinstance(provider, ast.Call)
            and isinstance(provider.func, ast.Name)
            and provider.func.id in self.uid_to_source_name
            and not provider.args
            and not provider.keywords
        ):
            return node
        values = self._sequence(provider.func.id)
        if values is None:
            return node
        comp.iter = ast.Tuple([ast.Constant(int(value)) for value in values], ast.Load())
        return ast.fix_missing_locations(node)


def _category_label(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _category_sequence_sort_key(row: tuple[Any, ...]) -> tuple[tuple[int, str], ...]:
    return tuple((0, "") if value is None else (1, str(value)) for value in row)


class FrozenCategorySequenceReductionNormalizer(ast.NodeTransformer):
    """Lower bounded category-provider sums through exact RunDomain cases.

    This bridge is the category analogue of ``FrozenIntegerSequenceReductionNormalizer``
    but it deliberately supports a provider whose *sequence* varies by policy.  The
    provider must already have an exact finite element domain in ``CategoryEncodingPlan``.
    Preparation evaluates it over the complete frozen RunDomain, records the exact
    finite sequence cases, and binds one small integer case code per policy.

    Each case is then handed to the existing ``UnrollLiteralReductions`` transform as
    a literal tuple.  The ordinary CategoryABI rewrite that runs later converts the
    substituted semantic string/None labels in native Cell calls to their int64 ABI
    codes.  Consequently no Python string, tuple or iterator enters native execution.

    The accepted execution shape is intentionally narrow: ``sum`` over one generator,
    no filters, one Name target, and a zero-argument Cell provider.  Vector/zip/filter
    reductions remain a separate I3B family.
    """

    def __init__(
        self,
        *,
        uid_to_source_name: Mapping[str, str],
        run_keys: Sequence[Any],
        space_instance: Callable[[Any], Any],
        register_point_input: Callable[
            [tuple[Any, ...], str, Callable[[Sequence[Any]], Any], str], str
        ],
        category_encoding: Any,
        max_slots: int = 16,
        max_cases: int = 16,
    ) -> None:
        self.uid_to_source_name = dict(uid_to_source_name)
        self.run_keys = tuple(run_keys)
        self.space_instance = space_instance
        self.register_point_input = register_point_input
        self.category_encoding = category_encoding
        self.max_slots = int(max_slots)
        self.max_cases = int(max_cases)

    @staticmethod
    def _generator(node: ast.AST) -> ast.GeneratorExp | None:
        if isinstance(node, ast.GeneratorExp):
            return node
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"list", "tuple"}
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.GeneratorExp)
        ):
            return node.args[0]
        return None

    def _rows(
        self, uid: str, keys: Sequence[Any], *, domain_labels: tuple[Any, ...]
    ) -> tuple[tuple[Any, ...], ...] | None:
        source_name = self.uid_to_source_name[uid]
        rows: list[tuple[Any, ...]] = []
        for key in keys:
            value = getattr(self.space_instance(key), source_name)()
            if not isinstance(value, (tuple, list)):
                return None
            row = tuple(value)
            if not row or len(row) > self.max_slots:
                return None
            if any(not _category_label(item) for item in row):
                return None
            outside = tuple(item for item in row if item not in domain_labels)
            if outside:
                raise StaticSequenceBridgeError(
                    f"category sequence provider {source_name!r} produced labels "
                    f"outside its proven CategoryABI domain: {outside!r}"
                )
            rows.append(row)
        return tuple(rows)

    @staticmethod
    def _unrolled_sum(node: ast.Call, gen: ast.GeneratorExp, row: tuple[Any, ...]) -> ast.AST:
        candidate = copy.deepcopy(node)
        candidate_gen = FrozenCategorySequenceReductionNormalizer._generator(candidate.args[0])
        if candidate_gen is None:
            raise StaticSequenceBridgeError("category reduction shape changed during unroll")
        candidate_gen.generators[0].iter = ast.Tuple(
            [ast.Constant(value) for value in row], ast.Load()
        )
        lowered = UnrollLiteralReductions().visit(candidate)
        return ast.fix_missing_locations(lowered)

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == "sum"
            and 1 <= len(node.args) <= 2
        ):
            return node
        gen = self._generator(node.args[0])
        if gen is None or len(gen.generators) != 1:
            return node
        comp = gen.generators[0]
        provider = comp.iter
        if (
            comp.ifs
            or comp.is_async
            or not isinstance(comp.target, ast.Name)
            or not isinstance(provider, ast.Call)
            or not isinstance(provider.func, ast.Name)
            or provider.args
            or provider.keywords
            or provider.func.id not in self.uid_to_source_name
        ):
            return node

        uid = provider.func.id
        did = self.category_encoding.sequence_result_domain_id(uid)
        if did is None:
            return node
        domain = self.category_encoding.domain(did)
        labels = tuple(domain.labels)
        if not labels or any(not _category_label(label) for label in labels):
            return node

        rows = self._rows(uid, self.run_keys, domain_labels=labels)
        if rows is None or not rows:
            return node
        cases = tuple(sorted(set(rows), key=_category_sequence_sort_key))
        if not cases or len(cases) > self.max_cases:
            return node

        case_exprs = [self._unrolled_sum(node, gen, row) for row in cases]
        if len(cases) == 1:
            return ast.copy_location(case_exprs[0], node)

        case_by_row = {row: index for index, row in enumerate(cases)}
        source_name = self.uid_to_source_name[uid]

        def case_getter(keys: Sequence[Any]) -> np.ndarray:
            current = self._rows(uid, keys, domain_labels=labels)
            if current is None:
                raise StaticSequenceBridgeError(
                    f"category sequence provider {source_name!r} changed shape/type"
                )
            codes: list[int] = []
            for row in current:
                if row not in case_by_row:
                    raise StaticSequenceBridgeError(
                        f"category sequence provider {source_name!r} produced unseen case {row!r}"
                    )
                codes.append(case_by_row[row])
            return np.asarray(codes, dtype=np.int64)

        token = (
            "static_category_sequence_reduction_case",
            uid,
            did,
            cases,
        )
        case_key = self.register_point_input(
            token,
            "int64",
            case_getter,
            f"frozen category sequence reduction case {source_name}",
        )
        selector = ast.Call(ast.Name("point_input", ast.Load()), [ast.Constant(case_key)], [])
        result = case_exprs[-1]
        for index in reversed(range(len(case_exprs) - 1)):
            result = ast.IfExp(
                test=ast.Compare(copy.deepcopy(selector), [ast.Eq()], [ast.Constant(index)]),
                body=case_exprs[index],
                orelse=result,
            )
        return ast.copy_location(ast.fix_missing_locations(result), node)


__all__ = [
    "FrozenSequenceAccessNormalizer",
    "FrozenSequenceMembershipNormalizer",
    "FrozenIntegerSequenceReductionNormalizer",
    "FrozenCategorySequenceReductionNormalizer",
    "StaticSequenceBridgeError",
]
