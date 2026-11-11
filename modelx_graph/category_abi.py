from __future__ import annotations

"""Backend-neutral deterministic finite-category ABI.

The semantic domain remains distinct from its physical int64 code.  This module has
no dependency on static graph/type analysis; frontends provide proven finite domains
and parameter positions.
"""

import ast
import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

class CategoryABIError(RuntimeError):
    pass


def _label_key(value: Any) -> tuple[int, str]:
    if value is None:
        return (0, "")
    if isinstance(value, str):
        return (1, value)
    raise CategoryABIError(
        f"finite-category ABI currently accepts only string/None labels, got {type(value).__name__}"
    )


def _ordered_labels(values: Sequence[Any]) -> tuple[Any, ...]:
    unique: dict[tuple[int, str], Any] = {}
    for value in values:
        unique[_label_key(value)] = value
    return tuple(unique[k] for k in sorted(unique))


def _label_token(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "none", "value": None}
    return {"type": "str", "value": value}


def _domain_id_for_ordered_labels(labels: Sequence[Any]) -> str:
    payload = json.dumps(
        [_label_token(x) for x in labels], sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"cat_{digest}"


@dataclass(frozen=True)
class CategoryDomain:
    domain_id: str
    labels: tuple[Any, ...]
    physical_dtype: str = "int64"
    provenance: tuple[str, ...] = ()

    @staticmethod
    def from_labels(labels: Sequence[Any], *provenance: str) -> "CategoryDomain":
        ordered = _ordered_labels(labels)
        if not ordered:
            raise CategoryABIError("finite-category domain is empty")
        return CategoryDomain(
            domain_id=_domain_id_for_ordered_labels(ordered), labels=ordered,
            provenance=tuple(sorted(set(x for x in provenance if x))),
        )

    def extended(self, labels: Sequence[Any], *provenance: str) -> "CategoryDomain":
        """Return a deterministic ABI-preserving finite-domain extension.

        Existing labels keep their physical integer codes.  Newly proven labels are
        appended in deterministic semantic order.  This differs intentionally from
        :meth:`from_labels`, whose canonical ordering is appropriate when a domain is
        created from scratch but would renumber existing labels during completion.
        """
        additions = _ordered_labels(tuple(x for x in labels if x not in self.labels))
        if not additions:
            return self
        ordered = tuple(self.labels) + tuple(additions)
        if len(set(ordered)) != len(ordered):
            raise CategoryABIError("finite-category domain extension contains duplicate labels")
        # Validate the complete sequence through the normal label contract.
        for value in ordered:
            _label_key(value)
        return CategoryDomain(
            domain_id=_domain_id_for_ordered_labels(ordered),
            labels=ordered,
            provenance=tuple(sorted(set((*self.provenance, *(x for x in provenance if x))))),
        )

    @property
    def code_by_label(self) -> Mapping[Any, int]:
        return {label: i for i, label in enumerate(self.labels)}

    def code(self, label: Any) -> int:
        try:
            return int(self.labels.index(label))
        except ValueError as exc:
            raise CategoryABIError(
                f"label {label!r} is outside frozen category domain {self.domain_id}"
            ) from exc

    def manifest(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "physical_dtype": self.physical_dtype,
            "labels": [_label_token(x) for x in self.labels],
            "label_code_map": [
                {"label": _label_token(label), "code": i}
                for i, label in enumerate(self.labels)
            ],
            "provenance": list(self.provenance),
            "reversible": True,
        }


@dataclass(frozen=True)
class CategoryTranslation:
    source_domain_id: str
    target_domain_id: str
    labels: tuple[Any, ...]
    code_map: tuple[tuple[int, int], ...]
    provenance: str = "semantic_label_subset"

    @property
    def identity(self) -> bool:
        return self.source_domain_id == self.target_domain_id and all(a == b for a, b in self.code_map)

    @property
    def affine_offset(self) -> int | None:
        if not self.code_map:
            return None
        offsets = {dst - src for src, dst in self.code_map}
        return next(iter(offsets)) if len(offsets) == 1 else None

    def manifest(self) -> dict[str, Any]:
        return {
            "source_domain_id": self.source_domain_id,
            "target_domain_id": self.target_domain_id,
            "labels": [_label_token(x) for x in self.labels],
            "code_map": [
                {"source_code": int(src), "target_code": int(dst)}
                for src, dst in self.code_map
            ],
            "identity": self.identity,
            "affine_offset": self.affine_offset,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class CategoryEncodingPlan:
    schema: str
    domains: Mapping[str, CategoryDomain]
    result_domains: Mapping[str, str]
    parameter_domains: Mapping[tuple[str, str], str]
    # Some Cells return a bounded tuple/list/set of semantic categories rather
    # than one category scalar.  The sequence itself remains a preparation/source
    # semantic object; this map describes only the element domain so generator
    # targets can carry category identity into native calls without Python strings.
    sequence_result_domains: Mapping[str, str] = None  # type: ignore[assignment]
    nullable_results: frozenset[str] = frozenset()
    nullable_parameters: frozenset[tuple[str, str]] = frozenset()
    nullable_sequence_results: frozenset[str] = frozenset()
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sequence_result_domains is None:
            object.__setattr__(self, "sequence_result_domains", {})

    @property
    def supported(self) -> bool:
        return not self.blockers

    def result_domain_id(self, uid: str) -> str | None:
        return self.result_domains.get(uid)

    def parameter_domain_id(self, uid: str, parameter: str) -> str | None:
        return self.parameter_domains.get((uid, parameter))

    def sequence_result_domain_id(self, uid: str) -> str | None:
        return self.sequence_result_domains.get(uid)

    def result_nullable(self, uid: str) -> bool:
        return uid in self.nullable_results

    def parameter_nullable(self, uid: str, parameter: str) -> bool:
        return (uid, parameter) in self.nullable_parameters

    def sequence_result_nullable(self, uid: str) -> bool:
        return uid in self.nullable_sequence_results

    def domain(self, domain_id: str) -> CategoryDomain:
        try:
            return self.domains[domain_id]
        except KeyError as exc:
            raise CategoryABIError(f"unknown category domain {domain_id!r}") from exc

    def translation(
        self, source_domain_id: str, target_domain_id: str, *,
        allowed_labels: Sequence[Any] | None = None,
    ) -> CategoryTranslation:
        source = self.domain(source_domain_id)
        target = self.domain(target_domain_id)
        labels = source.labels if allowed_labels is None else tuple(allowed_labels)
        if not labels:
            raise CategoryABIError(
                f"empty reachable category subset for {source_domain_id} -> {target_domain_id}"
            )
        if any(label not in source.labels for label in labels):
            raise CategoryABIError(
                f"category translation subset contains label outside {source_domain_id}: {labels!r}"
            )
        outside = tuple(label for label in labels if label not in target.labels)
        if outside:
            raise CategoryABIError(
                f"category translation {source_domain_id} -> {target_domain_id} has "
                f"reachable labels outside target domain: {outside!r}"
            )
        return CategoryTranslation(
            source_domain_id=source_domain_id, target_domain_id=target_domain_id,
            labels=labels,
            code_map=tuple((source.code(label), target.code(label)) for label in labels),
        )

    def rewrite_function(self, uid: str, fn: ast.FunctionDef) -> ast.FunctionDef:
        if self.blockers:
            raise CategoryABIError("category encoding plan is blocked: " + "; ".join(self.blockers))
        return _CategoryFormulaRewriter(self, uid, fn).rewrite()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "supported": self.supported,
            "domains": {k: self.domains[k].manifest() for k in sorted(self.domains)},
            "result_domains": dict(sorted(self.result_domains.items())),
            "parameter_domains": [
                {"uid": uid, "parameter": param, "domain_id": did}
                for (uid, param), did in sorted(self.parameter_domains.items())
            ],
            "sequence_result_domains": dict(sorted(self.sequence_result_domains.items())),
            "nullable_results": sorted(self.nullable_results),
            "nullable_parameters": [
                {"uid": uid, "parameter": param}
                for uid, param in sorted(self.nullable_parameters)
            ],
            "nullable_sequence_results": sorted(self.nullable_sequence_results),
            "used_translations": [
                row for row in getattr(self, "_used_translations", ())
            ],
            "blockers": list(self.blockers),
        }



class _CategoryFormulaRewriter(ast.NodeTransformer):
    def __init__(self, plan: CategoryEncodingPlan, uid: str, fn: ast.FunctionDef):
        self.plan = plan
        self.uid = uid
        self.fn = copy.deepcopy(fn)
        self.param_domains = {
            name: did for (cell_uid, name), did in plan.parameter_domains.items() if cell_uid == uid
        }
        self.nullable_params = {
            name for cell_uid, name in plan.nullable_parameters if cell_uid == uid
        }
        self.result_domain = plan.result_domains.get(uid)
        self.local_domains: dict[str, str] = {}
        self.nullable_locals: set[str] = set()
        # If a finite-result formula returns one local directly, that local carries
        # the result domain.  This captures guard-bearing point selectors without a
        # general object/local type system.
        if self.result_domain is not None:
            returned = {
                node.value.id
                for node in ast.walk(self.fn)
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
            }
            if len(returned) == 1:
                self.local_domains[next(iter(returned))] = self.result_domain
        self.active_param_labels: dict[str, tuple[Any, ...]] = {
            name: tuple(plan.domain(did).labels) for name, did in self.param_domains.items()
        }
        # Demand-directed typing below is deliberately limited to locals with
        # one source write.  Multiple writes require control-flow-sensitive
        # reaching-definition proof; treating every assignment to the same
        # Python local as one category value would be unsound.
        self.local_write_counts: dict[str, int] = {}
        for candidate in ast.walk(self.fn):
            if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, ast.Store):
                self.local_write_counts[candidate.id] = self.local_write_counts.get(candidate.id, 0) + 1
        # A local finite category may be introduced by ordinary Python source
        # rather than by a category-valued Cell. The important case is an
        # if-expression selecting between literal labels before forwarding the
        # local to a typed category parameter. Do not invent a category domain
        # from syntax alone: use only downstream category-ABI positions as
        # semantic demand, then prove the literal value set fits that domain.
        # The conditional itself stays an IfExp for Cython; only its semantic
        # labels are represented by stable int64 codes.
        self.local_category_demands = self._collect_local_category_demands()
        self.used_translations: list[CategoryTranslation] = []

    def _collect_local_category_demands(self) -> dict[str, tuple[str, ...]]:
        positions = getattr(self.plan, "_parameter_positions", {})
        demanded: dict[str, set[str]] = {}
        for node in ast.walk(self.fn):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            callee = node.func.id
            callee_positions = positions.get(callee, {})
            if not callee_positions:
                continue
            for (uid, parameter), did in self.plan.parameter_domains.items():
                if uid != callee:
                    continue
                pos = callee_positions.get(parameter)
                if pos is None or pos >= len(node.args):
                    continue
                arg = node.args[pos]
                if not isinstance(arg, ast.Name):
                    continue
                # Category parameters already have direct ABI authority. This
                # demand map exists only to type source locals.
                if arg.id in self.param_domains:
                    continue
                demanded.setdefault(arg.id, set()).add(did)
        return {
            name: tuple(sorted(domain_ids))
            for name, domain_ids in demanded.items()
        }

    @staticmethod
    def _literal_category_labels(node: ast.AST) -> tuple[Any, ...] | None:
        """Return the exact labels of a finite literal category expression."""
        if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, str)):
            return (node.value,)
        if isinstance(node, ast.IfExp):
            left = _CategoryFormulaRewriter._literal_category_labels(node.body)
            right = _CategoryFormulaRewriter._literal_category_labels(node.orelse)
            if left is None or right is None:
                return None
            return tuple(dict.fromkeys((*left, *right)))
        return None

    def _demanded_literal_local_domain(
        self, name: str, value: ast.AST,
    ) -> tuple[str, tuple[Any, ...]] | None:
        if self.local_write_counts.get(name, 0) != 1:
            return None
        labels = self._literal_category_labels(value)
        if not labels:
            return None
        demanded = list(self.local_category_demands.get(name, ()))
        # A local seeded from a proven result domain, for example a return-local,
        # is also authoritative for a later literal assignment.
        existing = self.local_domains.get(name)
        if existing is not None and existing not in demanded:
            demanded.append(existing)
        compatible = [
            did for did in demanded
            if all(label in self.plan.domain(did).labels for label in labels)
        ]
        if not compatible:
            return None
        # Prefer the narrowest proven domain. If the same local is later sent to
        # another compatible category ABI, visit_Call performs the existing
        # semantic code translation at that boundary.
        chosen = min(
            compatible,
            key=lambda did: (len(self.plan.domain(did).labels), did),
        )
        return chosen, labels

    def _expr_nullable(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.nullable_params or node.id in self.nullable_locals
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return self.plan.result_nullable(node.func.id)
        if isinstance(node, ast.IfExp):
            return self._expr_nullable(node.body) or self._expr_nullable(node.orelse)
        return False

    def _sequence_element_domain(self, node: ast.AST) -> tuple[str, bool] | None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            did = self.plan.sequence_result_domain_id(node.func.id)
            if did is not None:
                return did, self.plan.sequence_result_nullable(node.func.id)
        return None

    @staticmethod
    def _guard_stmt_terminal(st: ast.stmt) -> bool:
        if isinstance(st, (ast.Return, ast.Raise)):
            return True
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
            return isinstance(st.value.func, ast.Name) and st.value.func.id == "__guard_fail__"
        if isinstance(st, ast.If):
            return bool(st.orelse) and _CategoryFormulaRewriter._block_terminal(st.body) and _CategoryFormulaRewriter._block_terminal(st.orelse)
        return False

    @staticmethod
    def _block_terminal(body: Sequence[ast.stmt]) -> bool:
        return bool(body) and _CategoryFormulaRewriter._guard_stmt_terminal(body[-1])

    def _constraint(
        self, test: ast.AST, *, truth: bool,
    ) -> dict[str, tuple[Any, ...]] | None:
        """Return a local finite-label refinement for one simple guard.

        This is intentionally not a CFG. It accepts only equality/inequality and
        finite membership on one category parameter, which is enough to prove
        fallthrough after terminal source guards such as ``timing == 'EOM'``.
        """
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self._constraint(test.operand, truth=not truth)
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1):
            return None
        op = test.ops[0]
        left, right = test.left, test.comparators[0]
        if isinstance(right, ast.Name) and not isinstance(left, ast.Name):
            left, right = right, left
        if not isinstance(left, ast.Name) or self._expr_domain(left) is None:
            return None
        current = tuple(self.active_param_labels.get(left.id, ()))
        if not current:
            did = self._expr_domain(left)
            if did is not None:
                current = tuple(self.plan.domain(did).labels)
        if not current:
            return None
        selected: set[Any] | None = None
        if isinstance(right, ast.Constant) and (right.value is None or isinstance(right.value, str)):
            literal = right.value
            if isinstance(op, (ast.Eq, ast.Is)):
                selected = {literal} if truth else set(current) - {literal}
            elif isinstance(op, (ast.NotEq, ast.IsNot)):
                selected = set(current) - {literal} if truth else {literal}
        elif isinstance(right, (ast.Tuple, ast.List, ast.Set)) and isinstance(op, (ast.In, ast.NotIn)):
            labels: list[Any] = []
            for elt in right.elts:
                if not isinstance(elt, ast.Constant) or not (elt.value is None or isinstance(elt.value, str)):
                    return None
                labels.append(elt.value)
            pool = set(labels)
            in_truth = truth if isinstance(op, ast.In) else not truth
            selected = set(current) & pool if in_truth else set(current) - pool
        if selected is None:
            return None
        return {left.id: tuple(label for label in current if label in selected)}

    def _env_with_constraint(self, test: ast.AST, *, truth: bool) -> dict[str, tuple[Any, ...]]:
        env = dict(self.active_param_labels)
        update = self._constraint(test, truth=truth)
        if update:
            env.update(update)
        return env

    def _expr_labels(self, node: ast.AST) -> tuple[Any, ...] | None:
        if isinstance(node, ast.Name):
            did = self.param_domains.get(node.id) or self.local_domains.get(node.id)
            if did is None:
                return None
            if node.id in self.active_param_labels:
                return tuple(self.active_param_labels[node.id])
            return tuple(self.plan.domain(did).labels)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            did = self.plan.result_domains.get(node.func.id)
            return None if did is None else tuple(self.plan.domain(did).labels)
        if isinstance(node, ast.IfExp):
            did = self._expr_domain(node)
            if did is None:
                return None
            labels: list[Any] = []
            for branch in (node.body, node.orelse):
                values = self._expr_labels(branch)
                if values is not None:
                    labels.extend(values)
                elif (
                    isinstance(branch, ast.Constant)
                    and (branch.value is None or isinstance(branch.value, str))
                    and branch.value in self.plan.domain(did).labels
                ):
                    labels.append(branch.value)
                else:
                    return None
            return tuple(dict.fromkeys(labels))
        return None

    def _translated_value(
        self, node: ast.AST, source_domain_id: str, target_domain_id: str, *, context: str
    ) -> ast.AST:
        allowed = self._expr_labels(node)
        tr = self.plan.translation(
            source_domain_id, target_domain_id, allowed_labels=allowed
        )
        if tr.identity:
            return node
        self.used_translations.append(tr)
        offset = tr.affine_offset
        if offset is not None:
            if offset == 0:
                return node
            op: ast.operator = ast.Add() if offset > 0 else ast.Sub()
            translated = ast.BinOp(copy.deepcopy(node), op, ast.Constant(abs(int(offset))))
            if not self._expr_nullable(node):
                return ast.copy_location(translated, node)
            return ast.copy_location(ast.IfExp(
                test=ast.Compare(copy.deepcopy(node), [ast.Lt()], [ast.Constant(0)]),
                body=ast.Constant(-1),
                orelse=translated,
            ), node)
        # Domains are bounded to at most 16 labels.  Emit a compact label-code
        # translation only when it is genuinely non-affine.  The final branch is
        # unconditional because local/source proof establishes that ``node`` can
        # take only ``tr.labels`` at this program point.
        pairs = list(tr.code_map)
        result: ast.AST = ast.Constant(int(pairs[-1][1]))
        for src, dst in reversed(pairs[:-1]):
            result = ast.IfExp(
                test=ast.Compare(copy.deepcopy(node), [ast.Eq()], [ast.Constant(int(src))]),
                body=ast.Constant(int(dst)), orelse=result,
            )
        if self._expr_nullable(node):
            result = ast.IfExp(
                test=ast.Compare(copy.deepcopy(node), [ast.Lt()], [ast.Constant(0)]),
                body=ast.Constant(-1),
                orelse=result,
            )
        return ast.copy_location(ast.fix_missing_locations(result), node)

    def rewrite(self) -> ast.FunctionDef:
        out = self.visit(self.fn)
        assert isinstance(out, ast.FunctionDef)
        if self.used_translations:
            existing = list(getattr(self.plan, "_used_translations", ()))
            for tr in self.used_translations:
                row = tr.manifest()
                if row not in existing:
                    existing.append(row)
            object.__setattr__(self.plan, "_used_translations", tuple(existing))
        return ast.fix_missing_locations(out)

    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802
        prior = dict(self.active_param_labels)
        body: list[ast.stmt] = []
        for st in node.body:
            original_test = copy.deepcopy(st.test) if isinstance(st, ast.If) else None
            out = self.visit(st)
            if out is None:
                pass
            elif isinstance(out, list):
                body.extend(out)
            else:
                body.append(out)
            if (
                isinstance(st, ast.If) and not st.orelse and original_test is not None
                and self._block_terminal(st.body)
            ):
                self.active_param_labels = self._env_with_constraint(original_test, truth=False)
        node.body = body
        self.active_param_labels = prior
        return node

    def _expr_domain(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self.param_domains.get(node.id) or self.local_domains.get(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return self.plan.result_domains.get(node.func.id)
        if isinstance(node, ast.IfExp):
            left = self._expr_domain(node.body)
            right = self._expr_domain(node.orelse)
            if left is not None and left == right:
                return left
            if left is not None and right is None and isinstance(node.orelse, ast.Constant):
                if node.orelse.value in self.plan.domain(left).labels:
                    return left
            if right is not None and left is None and isinstance(node.body, ast.Constant):
                if node.body.value in self.plan.domain(right).labels:
                    return right
            return None
        return None

    def visit_Assign(self, node: ast.Assign):  # noqa: N802
        original_value = copy.deepcopy(node.value)
        domain_id = self._expr_domain(original_value)
        labels = self._expr_labels(original_value)
        nullable = self._expr_nullable(original_value)
        target_name = (
            node.targets[0].id
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            else None
        )
        if domain_id is None and target_name is not None:
            demanded = self._demanded_literal_local_domain(target_name, original_value)
            if demanded is not None:
                domain_id, labels = demanded
        if (
            domain_id is None and target_name is not None
            and target_name in self.local_domains
            and isinstance(original_value, ast.Call)
            and isinstance(original_value.func, ast.Name)
            and original_value.func.id == "point_input"
        ):
            # Direct point providers are bound to integer category inputs before
            # this rewrite.  The return-local seed in __init__ is the authority
            # linking that point_input to the Cell's result domain.
            domain_id = self.local_domains[target_name]
            labels = tuple(self.plan.domain(domain_id).labels)
        node = self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if domain_id is not None:
                node.value = self._encode_value(
                    node.value, domain_id, context=f"assign:{name}",
                    target_nullable=nullable,
                )
            if domain_id is None:
                self.local_domains.pop(name, None)
                self.active_param_labels.pop(name, None)
                self.nullable_locals.discard(name)
            else:
                self.local_domains[name] = domain_id
                self.active_param_labels[name] = (
                    tuple(self.plan.domain(domain_id).labels) if labels is None else tuple(labels)
                )
                if nullable:
                    self.nullable_locals.add(name)
                else:
                    self.nullable_locals.discard(name)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign):  # noqa: N802
        if node.value is None or not isinstance(node.target, ast.Name):
            return self.generic_visit(node)
        original_value = copy.deepcopy(node.value)
        domain_id = self._expr_domain(original_value)
        labels = self._expr_labels(original_value)
        nullable = self._expr_nullable(original_value)
        if domain_id is None:
            demanded = self._demanded_literal_local_domain(node.target.id, original_value)
            if demanded is not None:
                domain_id, labels = demanded
        node = self.generic_visit(node)
        name = node.target.id
        if domain_id is not None and node.value is not None:
            node.value = self._encode_value(
                node.value, domain_id, context=f"assign:{name}",
                target_nullable=nullable,
            )
        if domain_id is None:
            self.local_domains.pop(name, None)
            self.active_param_labels.pop(name, None)
            self.nullable_locals.discard(name)
        else:
            self.local_domains[name] = domain_id
            self.active_param_labels[name] = (
                tuple(self.plan.domain(domain_id).labels) if labels is None else tuple(labels)
            )
            if nullable:
                self.nullable_locals.add(name)
            else:
                self.nullable_locals.discard(name)
        return node

    def visit_GeneratorExp(self, node: ast.GeneratorExp):  # noqa: N802
        # Dynamic category propagation for one generator target over a Cell that
        # returns a bounded finite literal category sequence.  This carries only
        # the element semantic domain; it does not lower or execute the generator.
        if len(node.generators) != 1:
            return self.generic_visit(node)
        comp = node.generators[0]
        if not isinstance(comp.target, ast.Name):
            return self.generic_visit(node)
        element = self._sequence_element_domain(comp.iter)
        if element is None:
            return self.generic_visit(node)
        did, nullable = element
        name = comp.target.id
        prior_domain = self.local_domains.get(name)
        prior_labels = self.active_param_labels.get(name)
        prior_nullable = name in self.nullable_locals
        self.local_domains[name] = did
        self.active_param_labels[name] = tuple(self.plan.domain(did).labels)
        if nullable:
            self.nullable_locals.add(name)
        else:
            self.nullable_locals.discard(name)
        # Iterator is not itself a scalar category value. Rewrite it normally,
        # then rewrite filters/body under the generator-target category scope.
        comp.iter = self.visit(comp.iter)
        comp.ifs = [self.visit(x) for x in comp.ifs]
        node.elt = self.visit(node.elt)
        if prior_domain is None:
            self.local_domains.pop(name, None)
        else:
            self.local_domains[name] = prior_domain
        if prior_labels is None:
            self.active_param_labels.pop(name, None)
        else:
            self.active_param_labels[name] = prior_labels
        if prior_nullable:
            self.nullable_locals.add(name)
        else:
            self.nullable_locals.discard(name)
        return node

    def _encode_value(
        self, node: ast.AST, domain_id: str, *, context: str,
        target_nullable: bool = False,
    ) -> ast.AST:
        domain = self.plan.domain(domain_id)
        if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, str)):
            if node.value not in domain.labels:
                raise CategoryABIError(
                    f"{self.uid}:{context}: literal {node.value!r} outside {domain_id}"
                )
            return ast.copy_location(ast.Constant(domain.code(node.value)), node)
        if isinstance(node, ast.IfExp):
            return ast.copy_location(ast.IfExp(
                test=node.test,
                body=self._encode_value(
                    node.body, domain_id, context=context,
                    target_nullable=target_nullable,
                ),
                orelse=self._encode_value(
                    node.orelse, domain_id, context=context,
                    target_nullable=target_nullable,
                ),
            ), node)
        actual_domain = self._expr_domain(node)
        if actual_domain is not None:
            if actual_domain != domain_id:
                try:
                    return self._translated_value(
                        node, actual_domain, domain_id, context=context
                    )
                except CategoryABIError as exc:
                    raise CategoryABIError(
                        f"{self.uid}:{context}: incompatible category domains "
                        f"{actual_domain} vs {domain_id}: {exc}"
                    ) from exc
            return node
        # Provider-bound point_input already returns the frozen integer code for the
        # result domain.  The provider binder is responsible for installing the
        # matching semantic result-domain assignment before this rewrite.
        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "point_input"
        ):
            return node
        raise CategoryABIError(
            f"{self.uid}:{context}: dynamic value {ast.unparse(node)!r} has no category-domain proof"
        )

    def _category_use_error(self, node: ast.AST, context: str) -> CategoryABIError:
        return CategoryABIError(
            f"{self.uid}:{context}: semantic category value {ast.unparse(node)!r} "
            "cannot be used as an ordinary numeric/boolean value"
        )

    def visit_BinOp(self, node: ast.BinOp):  # noqa: N802
        if self._expr_domain(node.left) is not None:
            raise self._category_use_error(node.left, "binary_left")
        if self._expr_domain(node.right) is not None:
            raise self._category_use_error(node.right, "binary_right")
        return self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp):  # noqa: N802
        if self._expr_domain(node.operand) is not None:
            raise self._category_use_error(node.operand, "unary")
        return self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):  # noqa: N802
        for value in node.values:
            if self._expr_domain(value) is not None:
                raise self._category_use_error(value, "boolean")
        return self.generic_visit(node)

    def visit_If(self, node: ast.If):  # noqa: N802
        if self._expr_domain(node.test) is not None:
            raise self._category_use_error(node.test, "if_test")
        original_test = copy.deepcopy(node.test)
        node.test = self.visit(node.test)
        prior = dict(self.active_param_labels)
        self.active_param_labels = self._env_with_constraint(original_test, truth=True)
        out_body: list[ast.stmt] = []
        for st in node.body:
            rewritten = self.visit(st)
            if rewritten is None:
                continue
            if isinstance(rewritten, list):
                out_body.extend(rewritten)
            else:
                out_body.append(rewritten)
        node.body = out_body
        self.active_param_labels = dict(prior)
        self.active_param_labels = self._env_with_constraint(original_test, truth=False)
        out_else: list[ast.stmt] = []
        for st in node.orelse:
            rewritten = self.visit(st)
            if rewritten is None:
                continue
            if isinstance(rewritten, list):
                out_else.extend(rewritten)
            else:
                out_else.append(rewritten)
        node.orelse = out_else
        self.active_param_labels = prior
        return node

    def visit_IfExp(self, node: ast.IfExp):  # noqa: N802
        if self._expr_domain(node.test) is not None:
            raise self._category_use_error(node.test, "if_expression_test")
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Name):
            return node
        callee = node.func.id
        # Seed calls are positional after V02395F default/keyword materialization.
        param_names = [
            name for (uid, name), _did in self.plan.parameter_domains.items() if uid == callee
        ]
        if not param_names:
            return node
        # Parameter order is not stored in the plan; recover it from the callee
        # assignment ordering by requiring caller-side positions supplied by a map
        # attached during plan construction.  For safety the public builder adds
        # ``_parameter_positions`` below.
        positions = getattr(self.plan, "_parameter_positions", {}).get(callee, {})
        args = list(node.args)
        category_positions: dict[int, str] = {}
        for name, did in ((name, self.plan.parameter_domains[(callee, name)]) for name in param_names):
            pos = positions.get(name)
            if pos is None or pos >= len(args):
                raise CategoryABIError(f"{self.uid}: missing category call argument {callee}.{name}")
            category_positions[pos] = did
            args[pos] = self._encode_value(
                args[pos], did, context=f"call:{callee}:{name}",
                target_nullable=self.plan.parameter_nullable(callee, name),
            )
        for pos, arg in enumerate(args):
            actual = self._expr_domain(arg)
            if actual is not None and pos not in category_positions:
                raise self._category_use_error(arg, f"call:{callee}:arg{pos}")
        node.args = args
        return node

    def _comparison_pair(self, left: ast.AST, right: ast.AST):
        ldom = self._expr_domain(left)
        rdom = self._expr_domain(right)
        if ldom is not None and isinstance(right, ast.Constant) and (right.value is None or isinstance(right.value, str)):
            return left, right, ldom, False
        if rdom is not None and isinstance(left, ast.Constant) and (left.value is None or isinstance(left.value, str)):
            return right, left, rdom, True
        return None

    def visit_Compare(self, node: ast.Compare):  # noqa: N802
        node = self.generic_visit(node)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return node
        op = node.ops[0]
        right = node.comparators[0]

        pair = self._comparison_pair(node.left, right)
        if pair is not None and isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
            expr, literal, did, reversed_order = pair
            domain = self.plan.domain(did)
            if literal.value not in domain.labels:
                return ast.copy_location(
                    ast.Constant(isinstance(op, (ast.NotEq, ast.IsNot))), node
                )
            lhs: ast.AST = expr
            rhs: ast.AST = ast.Constant(domain.code(literal.value))
            if reversed_order:
                lhs, rhs = rhs, lhs
            native_op: ast.cmpop = ast.NotEq() if isinstance(op, (ast.NotEq, ast.IsNot)) else ast.Eq()
            return ast.copy_location(ast.Compare(lhs, [native_op], [rhs]), node)

        left_domain = self._expr_domain(node.left)
        right_domain = self._expr_domain(right)
        if (left_domain is not None or right_domain is not None) and not isinstance(op, (ast.In, ast.NotIn)):
            if left_domain is not None and right_domain is not None:
                if left_domain != right_domain:
                    raise CategoryABIError(
                        f"{self.uid}:compare: incompatible category domains "
                        f"{left_domain} vs {right_domain}"
                    )
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    return node
                raise self._category_use_error(node, "category_comparison")
            other = right if left_domain is not None else node.left
            if isinstance(other, ast.Constant) and not (other.value is None or isinstance(other.value, str)):
                if isinstance(op, (ast.Eq, ast.Is)):
                    return ast.copy_location(ast.Constant(False), node)
                if isinstance(op, (ast.NotEq, ast.IsNot)):
                    return ast.copy_location(ast.Constant(True), node)
            raise self._category_use_error(node, "category_comparison")

        # Membership over a category variable/result.  Labels outside the proven
        # domain can be discarded; an entirely disjoint membership folds exactly.
        did = self._expr_domain(node.left)
        if did is not None and isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.Tuple, ast.List, ast.Set)):
            domain = self.plan.domain(did)
            labels: list[Any] = []
            for elt in right.elts:
                if not isinstance(elt, ast.Constant) or not (elt.value is None or isinstance(elt.value, str)):
                    return node
                if elt.value in domain.labels:
                    labels.append(elt.value)
            if not labels:
                return ast.copy_location(ast.Constant(isinstance(op, ast.NotIn)), node)
            comparisons = [
                ast.Compare(
                    copy.deepcopy(node.left),
                    [ast.NotEq() if isinstance(op, ast.NotIn) else ast.Eq()],
                    [ast.Constant(domain.code(x))],
                )
                for x in labels
            ]
            if len(comparisons) == 1:
                return ast.copy_location(comparisons[0], node)
            return ast.copy_location(
                ast.BoolOp(
                    ast.And() if isinstance(op, ast.NotIn) else ast.Or(),
                    comparisons,
                ),
                node,
            )
        return node

    def _encode_return_value(self, node: ast.AST, domain_id: str) -> ast.AST:
        if isinstance(node, ast.IfExp):
            original_test = copy.deepcopy(node.test)
            test = self.visit(copy.deepcopy(node.test))
            prior = dict(self.active_param_labels)
            self.active_param_labels = self._env_with_constraint(original_test, truth=True)
            body = self._encode_return_value(copy.deepcopy(node.body), domain_id)
            self.active_param_labels = dict(prior)
            self.active_param_labels = self._env_with_constraint(original_test, truth=False)
            orelse = self._encode_return_value(copy.deepcopy(node.orelse), domain_id)
            self.active_param_labels = prior
            return ast.copy_location(ast.IfExp(test=test, body=body, orelse=orelse), node)
        visited = self.visit(copy.deepcopy(node))
        return self._encode_value(visited, domain_id, context="return")

    def visit_Return(self, node: ast.Return):  # noqa: N802
        if self.result_domain is not None and node.value is not None:
            return ast.copy_location(
                ast.Return(self._encode_return_value(node.value, self.result_domain)), node
            )
        return self.generic_visit(node)


def attach_parameter_positions(
    plan: CategoryEncodingPlan,
    parameter_names_by_uid: Mapping[str, Sequence[str]],
) -> CategoryEncodingPlan:
    positions = {
        uid: {name: pos for pos, name in enumerate(names)}
        for uid, names in parameter_names_by_uid.items()
    }
    object.__setattr__(plan, "_parameter_positions", positions)
    return plan


__all__ = [
    "CategoryABIError", "CategoryDomain", "CategoryTranslation", "CategoryEncodingPlan",
    "attach_parameter_positions",
]
