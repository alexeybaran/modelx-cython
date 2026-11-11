from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .sequential_ir import EvalOp, SequentialError, SequentialProgram
from .realized_trace import clear_realized_calculated


class LoopRecoveryError(SequentialError):
    pass


def _safe_equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        v = a == b
    except Exception:
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return False


@dataclass(frozen=True)
class ConstantBinding:
    value: Any = field(compare=False)

    def value_at(self, iteration: int) -> Any:
        return self.value

    @property
    def kind(self) -> str:
        return "constant"


@dataclass(frozen=True)
class AffineIntBinding:
    base: int
    step: int

    def value_at(self, iteration: int) -> int:
        return self.base + self.step * iteration

    @property
    def kind(self) -> str:
        return "affine_int"


@dataclass(frozen=True)
class PeriodicBinding:
    pattern: tuple[Any, ...] = field(compare=False)

    def value_at(self, iteration: int) -> Any:
        return self.pattern[iteration % len(self.pattern)]

    @property
    def kind(self) -> str:
        return "periodic"


@dataclass(frozen=True)
class InterleavedAffineIntBinding:
    """Several affine integer streams interleaved with a fixed exact period."""

    bases: tuple[int, ...]
    steps: tuple[int, ...]

    def value_at(self, iteration: int) -> int:
        period = len(self.bases)
        phase = iteration % period
        cycle = iteration // period
        return self.bases[phase] + self.steps[phase] * cycle

    @property
    def kind(self) -> str:
        return "interleaved_affine_int"


@dataclass(frozen=True)
class ArithmeticRunsIntBinding:
    """Exact integer sequence factored into arithmetic runs with one common step.

    This is a generic representation for flattened nested-loop/range coordinates
    and other piecewise arithmetic streams. Run boundaries are exact realized
    data; no meaning is assigned to the integer values.
    """

    run_ends: tuple[int, ...]
    run_starts: tuple[int, ...]
    step: int

    def value_at(self, iteration: int) -> int:
        run = bisect.bisect_right(self.run_ends, iteration)
        start_pos = 0 if run == 0 else self.run_ends[run - 1]
        return self.run_starts[run] + self.step * (iteration - start_pos)

    @property
    def kind(self) -> str:
        return "arithmetic_runs_int"


@dataclass(frozen=True)
class TableBinding:
    values: tuple[Any, ...] = field(compare=False)

    def value_at(self, iteration: int) -> Any:
        return self.values[iteration]

    @property
    def kind(self) -> str:
        return "table"


@dataclass(frozen=True)
class RunLengthBinding:
    """Exact run-length representation with a separately compressed run value stream."""

    run_ends: tuple[int, ...]
    run_values: Any = field(compare=False)

    def value_at(self, iteration: int) -> Any:
        run = bisect.bisect_right(self.run_ends, iteration)
        return self.run_values.value_at(run)

    @property
    def kind(self) -> str:
        return "run_length"


Binding = (
    ConstantBinding
    | AffineIntBinding
    | PeriodicBinding
    | InterleavedAffineIntBinding
    | ArithmeticRunsIntBinding
    | TableBinding
    | RunLengthBinding
)


def binding_payload_items(binding: Binding) -> int:
    """Approximate exact data items retained by one binding representation."""
    if isinstance(binding, ConstantBinding):
        return 1
    if isinstance(binding, AffineIntBinding):
        return 2
    if isinstance(binding, PeriodicBinding):
        return len(binding.pattern)
    if isinstance(binding, InterleavedAffineIntBinding):
        return 2 * len(binding.bases)
    if isinstance(binding, ArithmeticRunsIntBinding):
        return len(binding.run_ends) + len(binding.run_starts) + 1
    if isinstance(binding, RunLengthBinding):
        return len(binding.run_ends) + binding_payload_items(binding.run_values)
    return len(binding.values)


def _periodic_binding(values: Sequence[Any], *, max_period: int = 64) -> PeriodicBinding | None:
    n = len(values)
    # Require at least two repeats. The first exact period wins and is therefore
    # also the smallest payload among the tested periods.
    for period in range(2, min(max_period, n // 2) + 1):
        if all(_safe_equal(values[i], values[i % period]) for i in range(period, n)):
            return PeriodicBinding(tuple(values[:period]))
    return None


def _interleaved_affine_binding(
    values: Sequence[Any], *, max_period: int = 64
) -> InterleavedAffineIntBinding | None:
    if not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in values):
        return None
    ints = [int(v) for v in values]
    n = len(ints)
    for period in range(2, min(max_period, n // 2) + 1):
        if n < 2 * period:
            break
        bases = tuple(ints[r] for r in range(period))
        steps = tuple(ints[r + period] - ints[r] for r in range(period))
        ok = True
        for i, value in enumerate(ints):
            phase = i % period
            cycle = i // period
            if value != bases[phase] + steps[phase] * cycle:
                ok = False
                break
        if ok:
            return InterleavedAffineIntBinding(bases, steps)
    return None




def _arithmetic_runs_int_binding(values: Sequence[Any]) -> ArithmeticRunsIntBinding | None:
    """Factor an integer stream into exact arithmetic runs sharing one step.

    The step is selected solely by adjacent-difference support. This catches
    flattened nested ranges and rolling windows without assuming which argument
    is time, segment, policy, or any other model-specific coordinate.
    """
    if len(values) < 4 or not all(
        isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
        for v in values
    ):
        return None
    ints = [int(v) for v in values]
    diffs = Counter(ints[i] - ints[i - 1] for i in range(1, len(ints)))
    if not diffs:
        return None
    # Deterministic: strongest-supported step, then smaller absolute step/value.
    step, support = min(
        diffs.items(), key=lambda kv: (-kv[1], abs(kv[0]), kv[0])
    )
    # A run representation should explain a material amount of adjacency.
    if support < max(2, len(ints) // 4):
        return None

    run_ends: list[int] = []
    run_starts: list[int] = [ints[0]]
    for i in range(1, len(ints)):
        if ints[i] - ints[i - 1] != step:
            run_ends.append(i)
            run_starts.append(ints[i])
    run_ends.append(len(ints))
    return ArithmeticRunsIntBinding(tuple(run_ends), tuple(run_starts), step)

def _infer_binding(values: Sequence[Any], *, allow_run_length: bool) -> Binding:
    if not values:
        raise LoopRecoveryError("cannot bind an empty value sequence")
    first = values[0]
    if all(_safe_equal(first, v) for v in values[1:]):
        return ConstantBinding(first)

    candidates: list[Binding] = [TableBinding(tuple(values))]
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in values):
        ints = [int(v) for v in values]
        if len(ints) == 1:
            return ConstantBinding(ints[0])
        step = ints[1] - ints[0]
        if all(ints[i] == ints[0] + step * i for i in range(len(ints))):
            candidates.append(AffineIntBinding(ints[0], step))

    periodic = _periodic_binding(values)
    if periodic is not None:
        candidates.append(periodic)
    interleaved = _interleaved_affine_binding(values)
    if interleaved is not None:
        candidates.append(interleaved)
    arithmetic_runs = _arithmetic_runs_int_binding(values)
    if arithmetic_runs is not None:
        candidates.append(arithmetic_runs)

    if allow_run_length and len(values) >= 4:
        run_values = [first]
        run_ends = []
        for i in range(1, len(values)):
            if not _safe_equal(values[i], run_values[-1]):
                run_ends.append(i)
                run_values.append(values[i])
        run_ends.append(len(values))
        if len(run_values) < len(values):
            value_binding = _infer_binding(run_values, allow_run_length=False)
            candidates.append(RunLengthBinding(tuple(run_ends), value_binding))

    # Deterministic tie order prefers semantically simpler bindings in the order
    # candidates were appended. This only changes representation, never values.
    return min(enumerate(candidates), key=lambda x: (binding_payload_items(x[1]), x[0]))[1]


def infer_binding(values: Sequence[Any]) -> Binding:
    return _infer_binding(values, allow_run_length=True)


@dataclass(frozen=True)
class EvalTemplate:
    schema_uid: str
    space_family_uid: str
    obj_binding: Binding = field(compare=False)
    arg_bindings: tuple[Binding, ...] = field(compare=False)

    def runtime_node(self, occurrence: int) -> tuple[Any, tuple[Any, ...]]:
        obj = self.obj_binding.value_at(occurrence)
        key = tuple(binding.value_at(occurrence) for binding in self.arg_bindings)
        return obj, key

    @property
    def role(self) -> tuple[str, str, int]:
        return (self.schema_uid, self.space_family_uid, len(self.arg_bindings))


@dataclass(frozen=True)
class LiteralBlock:
    ops: tuple[EvalOp, ...]

    @property
    def operation_count(self) -> int:
        return len(self.ops)

    @property
    def stored_body_ops(self) -> int:
        return len(self.ops)


@dataclass(frozen=True)
class LoopBlock:
    start: int
    body_size: int
    repetitions: int
    body: tuple[EvalTemplate, ...]
    # Dependency regimes are descriptive metadata only in realized Stage 2.
    # They no longer duplicate an otherwise identical instruction body.
    dependency_variant_ids: tuple[int, ...] = field(compare=False, default=())
    dependency_variant_count: int = field(compare=False, default=0)
    source_node_ids: tuple[int, ...] = field(compare=False, default=())

    @property
    def operation_count(self) -> int:
        return self.body_size * self.repetitions

    @property
    def stored_body_ops(self) -> int:
        return self.body_size

    @property
    def saved_ops(self) -> int:
        return self.operation_count - self.stored_body_ops

    def expand_runtime_nodes(self) -> tuple[tuple[Any, tuple[Any, ...]], ...]:
        out = []
        for i in range(self.repetitions):
            for op in self.body:
                out.append(op.runtime_node(i))
        return tuple(out)


@dataclass(frozen=True)
class PhaseGrammar:
    """RePair-style shared grammar for exact phase instruction shapes.

    Terminals are formula roles. Non-terminals are binary rules. Variant programs
    reference terminals/non-terminals. This shares repeated subsequences across
    different realized branch/phase shapes without inventing unseen control flow.
    """

    terminal_roles: tuple[tuple[str, str, int], ...]
    rules: tuple[tuple[int, int], ...]
    variant_programs: tuple[tuple[int, ...], ...]

    @property
    def terminal_count(self) -> int:
        return len(self.terminal_roles)

    @property
    def stored_symbols(self) -> int:
        return sum(len(p) for p in self.variant_programs) + 2 * len(self.rules)

    def _expand_symbol(self, symbol: int, out: list[int]) -> None:
        if symbol < self.terminal_count:
            out.append(symbol)
            return
        ridx = symbol - self.terminal_count
        left, right = self.rules[ridx]
        self._expand_symbol(left, out)
        self._expand_symbol(right, out)

    def expand_variant(self, phase_id: int) -> tuple[int, ...]:
        out: list[int] = []
        for symbol in self.variant_programs[phase_id]:
            self._expand_symbol(symbol, out)
        return tuple(out)

    def direct_terminal_counts(self) -> Counter[int]:
        result: Counter[int] = Counter()
        for program in self.variant_programs:
            for symbol in program:
                if symbol < self.terminal_count:
                    result[symbol] += 1
        for left, right in self.rules:
            if left < self.terminal_count:
                result[left] += 1
            if right < self.terminal_count:
                result[right] += 1
        return result


@dataclass(frozen=True)
class PhaseLoopBlock:
    """Exact outer loop with observed phase table and shared grammar.

    Each formula role has one binding template across all of its concrete
    occurrences in the phase region. The grammar determines only the observed
    role order. Exact per-role binding tables reconstruct the concrete runtime
    nodes, so irregular argument values and dependency regimes remain valid.
    """

    start: int
    stop: int
    anchor_role: tuple[str, str, int]
    phase_ids: tuple[int, ...]
    grammar: PhaseGrammar
    role_templates: tuple[EvalTemplate, ...]

    @property
    def repetitions(self) -> int:
        return len(self.phase_ids)

    @property
    def phase_count(self) -> int:
        return len(self.grammar.variant_programs)

    @property
    def operation_count(self) -> int:
        lengths = [len(self.grammar.expand_variant(i)) for i in range(self.phase_count)]
        return sum(lengths[pid] for pid in self.phase_ids)

    @property
    def stored_body_ops(self) -> int:
        # Formula-role binding templates plus shared structural grammar.
        return len(self.role_templates) + self.grammar.stored_symbols

    @property
    def saved_ops(self) -> int:
        return self.operation_count - self.stored_body_ops

    @property
    def grammar_terminal_callsite_max(self) -> int:
        counts = self.grammar.direct_terminal_counts()
        return max(counts.values(), default=0)

    def expand_runtime_nodes(self) -> tuple[tuple[Any, tuple[Any, ...]], ...]:
        counters = [0] * len(self.role_templates)
        expanded = [self.grammar.expand_variant(i) for i in range(self.phase_count)]
        out: list[tuple[Any, tuple[Any, ...]]] = []
        for pid in self.phase_ids:
            for role_id in expanded[pid]:
                occ = counters[role_id]
                out.append(self.role_templates[role_id].runtime_node(occ))
                counters[role_id] += 1
        return tuple(out)


@dataclass(frozen=True)
class LoopFamilyTemplate:
    """Shared code template for structurally related exact loop instances.

    Members keep their own concrete bindings and control streams. The family
    factors only repeated formula-role code across disjoint realized regions.
    """

    member_block_indices: tuple[int, ...]
    grammar: PhaseGrammar

    @property
    def role_count(self) -> int:
        return self.grammar.terminal_count

    @property
    def variant_count(self) -> int:
        return len(self.grammar.variant_programs)

    @property
    def stored_code_ops(self) -> int:
        return self.grammar.terminal_count + self.grammar.stored_symbols


@dataclass(frozen=True)
class FormulaOp:
    """One globally interned formula operation.

    A FormulaOp is code identity only. Concrete Space objects and arguments are
    deliberately absent; they belong to per-loop binding streams.  This is the
    key Stage-2 normalization that prevents one formula from being duplicated in
    executable syntax simply because it is evaluated at many realized arguments.
    """

    op_id: int
    schema_uid: str
    space_family_uid: str
    arity: int
    name: str
    fullname: str

    @property
    def role(self) -> tuple[str, str, int]:
        return (self.schema_uid, self.space_family_uid, self.arity)


@dataclass(frozen=True)
class RoleBindingTemplate:
    """Exact runtime-object/argument stream for one FormulaOp in one loop instance."""

    formula_op_id: int
    obj_binding: Binding = field(compare=False)
    arg_bindings: tuple[Binding, ...] = field(compare=False)

    def runtime_node(self, occurrence: int) -> tuple[Any, tuple[Any, ...]]:
        obj = self.obj_binding.value_at(occurrence)
        key = tuple(binding.value_at(occurrence) for binding in self.arg_bindings)
        return obj, key


@dataclass(frozen=True)
class LoopCodeFamily:
    """Executable shared loop code.

    Grammar terminals point to globally interned FormulaOps.  Member instances
    contain only exact bindings and an observed variant/control stream.
    """

    family_id: int
    formula_op_ids: tuple[int, ...]
    grammar: PhaseGrammar

    @property
    def is_atomic_repeat(self) -> bool:
        # A one-formula family needs no formula-specific loop body. It is executed
        # by the global REPEAT_FORMULA primitive with formula ID + bindings as data.
        return len(self.formula_op_ids) == 1

    @property
    def stored_code_ops(self) -> int:
        # Formula definitions are global. Atomic one-formula families use one
        # generic repeat primitive and therefore add only control/grammar data.
        return self.grammar.stored_symbols if self.is_atomic_repeat else len(self.formula_op_ids) + self.grammar.stored_symbols

    def expanded_variants(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self.grammar.expand_variant(i) for i in range(len(self.grammar.variant_programs)))


@dataclass(frozen=True)
class LoopInstancePlan:
    """One exact occurrence of a shared LoopCodeFamily in execution order."""

    block_index: int
    family_id: int
    variant_ids: tuple[int, ...]
    role_bindings: tuple[RoleBindingTemplate | None, ...]


@dataclass(frozen=True)
class CanonicalExecutionPlan:
    """Final Stage-2 executable normalization.

    Formula identity, reusable loop code and concrete binding/control data are
    separate. This is the representation future Python/Cython codegen should
    lower directly.
    """

    formula_ops: tuple[FormulaOp, ...]
    code_families: tuple[LoopCodeFamily, ...]
    loop_instances: tuple[LoopInstancePlan, ...]
    block_to_instance: tuple[int | None, ...]

    @property
    def formula_op_count(self) -> int:
        return len(self.formula_ops)

    def formula_code_site_counts(self) -> Counter[int]:
        result: Counter[int] = Counter()
        atomic_ops: set[int] = set()
        for family in self.code_families:
            if family.is_atomic_repeat:
                atomic_ops.update(family.formula_op_ids)
                continue
            for op_id in set(family.formula_op_ids):
                result[op_id] += 1
        # A formula used only by atomic repeats has one global generic callsite.
        # If it is already present in a non-atomic family, the atomic repeat passes
        # the FormulaOp ID as data and adds no formula-specific syntax.
        for op_id in atomic_ops:
            if result[op_id] == 0:
                result[op_id] = 1
        return result

    def formula_name_code_site_counts(self) -> Counter[str]:
        by_id = {op.op_id: op for op in self.formula_ops}
        result: Counter[str] = Counter()
        for op_id, count in self.formula_code_site_counts().items():
            result[by_id[op_id].name] += count
        return result

    @property
    def stored_code_ops(self) -> int:
        atomic_primitive = 1 if any(f.is_atomic_repeat for f in self.code_families) else 0
        return len(self.formula_ops) + atomic_primitive + sum(f.stored_code_ops for f in self.code_families)

    def render_syntax(self) -> str:
        """Render compact code-shaped Stage-2 syntax for audits/codegen design.

        Formula names/fullnames are declared exactly once. Loop families and
        instances reference numeric FormulaOp IDs, so repeated realized arguments
        cannot duplicate variable names in the stored evaluation syntax.
        """
        lines = ["FORMULAS"]
        for op in self.formula_ops:
            lines.append(f"  F{op.op_id} = {op.fullname} / arity={op.arity}")
        lines.append("LOOP_FAMILIES")
        for family in self.code_families:
            ops = ",".join(f"F{x}" for x in family.formula_op_ids)
            kind = "ATOMIC_REPEAT" if family.is_atomic_repeat else "FAMILY"
            lines.append(
                f"  {kind} {family.family_id}: terminals=[{ops}] "
                f"variants={len(family.grammar.variant_programs)} rules={len(family.grammar.rules)}"
            )
        lines.append("LOOP_INSTANCES")
        for inst in self.loop_instances:
            lines.append(
                f"  BLOCK {inst.block_index} -> FAMILY {inst.family_id} "
                f"iterations={len(inst.variant_ids)}"
            )
        return "\n".join(lines)


Block = LiteralBlock | LoopBlock | PhaseLoopBlock


@dataclass
class StructuredSequentialProgram:
    sequential: SequentialProgram = field(repr=False)
    blocks: tuple[Block, ...]
    recovery_mode: str = "local"
    loop_families: tuple[LoopFamilyTemplate, ...] = ()
    canonical_plan: CanonicalExecutionPlan | None = None
    exact_expansion_validated: bool = False

    @property
    def loop_count(self) -> int:
        return sum(isinstance(b, (LoopBlock, PhaseLoopBlock)) for b in self.blocks)

    @property
    def local_loop_count(self) -> int:
        return sum(isinstance(b, LoopBlock) for b in self.blocks)

    @property
    def phase_loop_count(self) -> int:
        return sum(isinstance(b, PhaseLoopBlock) for b in self.blocks)

    @property
    def phase_variant_count(self) -> int:
        return sum(b.phase_count for b in self.blocks if isinstance(b, PhaseLoopBlock))

    @property
    def operation_count(self) -> int:
        return sum(b.operation_count for b in self.blocks)

    @property
    def stored_body_ops(self) -> int:
        return sum(b.stored_body_ops for b in self.blocks)

    @property
    def compression_ratio(self) -> float:
        stored = self.stored_body_ops
        return self.operation_count / stored if stored else 1.0

    @property
    def control_items(self) -> int:
        # Phase IDs are compact control metadata, not duplicated formula bodies.
        return sum(len(b.phase_ids) for b in self.blocks if isinstance(b, PhaseLoopBlock))

    def iter_bindings(self):
        """Yield every exact binding stored by recovered loop templates."""
        for block in self.blocks:
            templates = ()
            if isinstance(block, LoopBlock):
                templates = block.body
            elif isinstance(block, PhaseLoopBlock):
                templates = block.role_templates
            for template in templates:
                yield template.obj_binding
                yield from template.arg_bindings

    @property
    def binding_payload_items(self) -> int:
        return sum(binding_payload_items(binding) for binding in self.iter_bindings())

    def binding_kind_counts(self) -> Counter[str]:
        return Counter(binding.kind for binding in self.iter_bindings())

    @property
    def representation_items(self) -> int:
        """Approximate structural + control + exact-binding representation size."""
        return self.stored_body_ops + self.control_items + self.binding_payload_items

    @property
    def representation_compression_ratio(self) -> float:
        items = self.representation_items
        return self.operation_count / items if items else 1.0

    @property
    def family_code_ops(self) -> int:
        """Stored structural code after sharing related loop-family templates."""
        loop_member_ids = {i for f in self.loop_families for i in f.member_block_indices}
        literals = sum(
            b.stored_body_ops for i, b in enumerate(self.blocks)
            if isinstance(b, LiteralBlock) or (i not in loop_member_ids and isinstance(b, (LoopBlock, PhaseLoopBlock)))
        )
        family_code = 0
        for family in self.loop_families:
            if len(family.member_block_indices) == 1:
                family_code += self.blocks[family.member_block_indices[0]].stored_body_ops
            else:
                family_code += family.stored_code_ops
        return literals + family_code

    @property
    def family_code_compression_ratio(self) -> float:
        return self.operation_count / self.family_code_ops if self.family_code_ops else 1.0

    def family_role_counts(self) -> Counter[tuple[str, str, int]]:
        result: Counter[tuple[str, str, int]] = Counter()
        for family in self.loop_families:
            for role in family.grammar.terminal_roles:
                result[role] += 1
        return result

    def executable_formula_site_counts(self) -> Counter[tuple[str, str, int]]:
        """Count physical formula sites in the actual canonical loop code."""
        if self.canonical_plan is None:
            return self.family_role_counts()
        by_id = {op.op_id: op for op in self.canonical_plan.formula_ops}
        result: Counter[tuple[str, str, int]] = Counter()
        for op_id, count in self.canonical_plan.formula_code_site_counts().items():
            result[by_id[op_id].role] += count
        return result

    def executable_formula_name_site_counts(self) -> Counter[str]:
        if self.canonical_plan is None:
            # Fall back to schema names from the realized graph.
            role_name = {}
            for n in self.sequential.trace.nodes:
                role_name.setdefault(n.shape_token, n.schema.name)
            result: Counter[str] = Counter()
            for role, count in self.family_role_counts().items():
                result[role_name.get(role, role[0])] += count
            return result
        return self.canonical_plan.formula_name_code_site_counts()

    @property
    def canonical_code_ops(self) -> int:
        if self.canonical_plan is None:
            return self.family_code_ops
        # Literal concrete operations are boundary/setup schedule, not recurring
        # formula-code definitions. Count them separately from reusable loop code.
        literals = sum(b.stored_body_ops for b in self.blocks if isinstance(b, LiteralBlock))
        return literals + self.canonical_plan.stored_code_ops

    @property
    def canonical_code_compression_ratio(self) -> float:
        return self.operation_count / self.canonical_code_ops if self.canonical_code_ops else 1.0

    def render_canonical_syntax(self) -> str:
        """Render complete code-shaped schedule using global FormulaOp IDs.

        Fully qualified variable/formula names occur only in the FORMULAS registry.
        Literal operations and loop families reference F<n>, so argument/time
        variation cannot replicate source-level variable names.
        """
        if self.canonical_plan is None:
            raise LoopRecoveryError("canonical execution plan is unavailable")
        plan = self.canonical_plan
        lines = [plan.render_syntax(), "EXECUTION"]
        role_to_op = {op.role: op.op_id for op in plan.formula_ops}
        by_id = self.sequential.node_by_id
        for block_index, block in enumerate(self.blocks):
            if isinstance(block, LiteralBlock):
                ids = [role_to_op[by_id[op.node_id].shape_token] for op in block.ops]
                lines.append(
                    f"  LITERAL {block_index}: " + ",".join(f"F{x}" for x in ids)
                )
            else:
                instance_index = plan.block_to_instance[block_index]
                if instance_index is None:
                    raise LoopRecoveryError("canonical plan missing loop instance")
                inst = plan.loop_instances[instance_index]
                lines.append(
                    f"  LOOP {block_index}: FAMILY {inst.family_id} "
                    f"iterations={len(inst.variant_ids)}"
                )
        return "\n".join(lines)

    def expand_runtime_nodes(self) -> tuple[tuple[Any, tuple[Any, ...]], ...]:
        by_id = self.sequential.node_by_id
        out = []
        for block in self.blocks:
            if isinstance(block, LiteralBlock):
                out.extend(by_id[op.node_id].runtime_node for op in block.ops)
            else:
                out.extend(block.expand_runtime_nodes())
        return tuple(out)

    def iter_canonical_runtime_nodes(self):
        """Yield exact runtime nodes from the canonical Stage-2 plan without materializing the trace."""
        if self.canonical_plan is None:
            by_id = self.sequential.node_by_id
            for block in self.blocks:
                if isinstance(block, LiteralBlock):
                    for op in block.ops:
                        yield by_id[op.node_id].runtime_node
                else:
                    yield from block.expand_runtime_nodes()
            return

        plan = self.canonical_plan
        by_id = self.sequential.node_by_id
        for block_index, block in enumerate(self.blocks):
            if isinstance(block, LiteralBlock):
                for op in block.ops:
                    yield by_id[op.node_id].runtime_node
                continue
            instance_index = plan.block_to_instance[block_index]
            if instance_index is None:
                raise LoopRecoveryError("canonical plan missing loop instance")
            instance = plan.loop_instances[instance_index]
            family = plan.code_families[instance.family_id]
            expanded = family.expanded_variants()
            counters = [0] * len(family.formula_op_ids)
            for variant_id in instance.variant_ids:
                for terminal_id in expanded[variant_id]:
                    binding = instance.role_bindings[terminal_id]
                    if binding is None:
                        raise LoopRecoveryError("canonical family requested unbound role")
                    yield binding.runtime_node(counters[terminal_id])
                    counters[terminal_id] += 1

    def expand_canonical_runtime_nodes(self) -> tuple[tuple[Any, tuple[Any, ...]], ...]:
        return tuple(self.iter_canonical_runtime_nodes())

    def validate_exact_expansion(self) -> None:
        expected = self.sequential.runtime_nodes()
        actual = self.expand_runtime_nodes()
        if len(expected) != len(actual):
            raise LoopRecoveryError(f"loop expansion length mismatch {len(actual)} != {len(expected)}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            if a[0] is not b[0] or a[1] != b[1]:
                raise LoopRecoveryError(f"loop expansion differs at operation {i}: {a!r} != {b!r}")

        canonical = self.expand_canonical_runtime_nodes()
        if len(expected) != len(canonical):
            raise LoopRecoveryError(
                f"canonical expansion length mismatch {len(canonical)} != {len(expected)}"
            )
        for i, (a, b) in enumerate(zip(canonical, expected)):
            if a[0] is not b[0] or a[1] != b[1]:
                raise LoopRecoveryError(
                    f"canonical loop-family expansion differs at operation {i}: {a!r} != {b!r}"
                )
        self.exact_expansion_validated = True

    def execute(self, *, clear_first: bool = True, validate: bool = False, direct: bool = True):
        if not self.exact_expansion_validated:
            self.validate_exact_expansion()
        if clear_first:
            if direct:
                self.sequential._clear_scheduled_values()
            else:
                clear_realized_calculated(self.sequential.trace.model)

        system = self.sequential.trace.model._impl.system
        raw_events = None
        if validate:
            with system.trace_stack(maxlen=None):
                self._execute_blocks(direct=direct)
                result = self.sequential._read_targets()
                raw_events = list(system.callstack.tracestack)
        else:
            self._execute_blocks(direct=direct)
            result = self.sequential._read_targets()

        if validate:
            replay = self.sequential._validate_replay(raw_events or [], direct=direct, result=result)
            if not replay.ok:
                raise LoopRecoveryError("structured replay diverged from realized schedule: " + "; ".join(replay.reasons))
            return result, replay
        return result

    def _eval_runtime_node(self, obj: Any, key: tuple[Any, ...], *, direct: bool) -> None:
        if direct:
            if key in obj.input_keys:
                raise LoopRecoveryError("scheduled formula changed Cell value into explicit input")
            obj.on_eval_formula(key)
        else:
            obj.get_value_from_key(key)

    def _execute_blocks(self, *, direct: bool) -> None:
        by_id = self.sequential.node_by_id
        plan = self.canonical_plan
        for block_index, block in enumerate(self.blocks):
            if isinstance(block, LiteralBlock):
                for op in block.ops:
                    node = by_id[op.node_id]
                    self._eval_runtime_node(node.obj, node.args, direct=direct)
                continue

            if plan is not None:
                instance_index = plan.block_to_instance[block_index]
                if instance_index is None:
                    raise LoopRecoveryError("canonical plan missing loop instance")
                instance = plan.loop_instances[instance_index]
                family = plan.code_families[instance.family_id]
                counters = [0] * len(family.formula_op_ids)
                expanded = family.expanded_variants()
                for variant_id in instance.variant_ids:
                    for terminal_id in expanded[variant_id]:
                        binding = instance.role_bindings[terminal_id]
                        if binding is None:
                            raise LoopRecoveryError("loop family requested unbound formula role")
                        occ = counters[terminal_id]
                        self._eval_runtime_node(*binding.runtime_node(occ), direct=direct)
                        counters[terminal_id] += 1
                continue

            # Legacy exact executor retained as a defensive fallback.
            if isinstance(block, LoopBlock):
                for i in range(block.repetitions):
                    for tmpl in block.body:
                        self._eval_runtime_node(*tmpl.runtime_node(i), direct=direct)
            else:
                counters = [0] * len(block.role_templates)
                expanded = [block.grammar.expand_variant(i) for i in range(block.phase_count)]
                for pid in block.phase_ids:
                    for role_id in expanded[pid]:
                        occ = counters[role_id]
                        self._eval_runtime_node(
                            *block.role_templates[role_id].runtime_node(occ), direct=direct
                        )
                        counters[role_id] += 1

    def loop_role_counts(self) -> Counter[tuple[str, str, int]]:
        """Count how many recovered loop templates reference each formula role.

        Literal boundary operations are intentionally excluded. This is the
        canonical-code quality metric: a role used throughout one projection
        loop counts once even if it executes thousands of times.
        """
        result: Counter[tuple[str, str, int]] = Counter()
        for block in self.blocks:
            if isinstance(block, LoopBlock):
                for role in {t.role for t in block.body}:
                    result[role] += 1
            elif isinstance(block, PhaseLoopBlock):
                for role in {t.role for t in block.role_templates}:
                    result[role] += 1
        return result

    def stored_role_counts(self) -> Counter[tuple[str, str, int]]:
        """Count formula-role appearances in the stored Stage-2 representation."""
        result: Counter[tuple[str, str, int]] = Counter()
        by_id = self.sequential.node_by_id
        for block in self.blocks:
            if isinstance(block, LiteralBlock):
                for op in block.ops:
                    n = by_id[op.node_id]
                    result[(n.schema_uid, n.space_family_uid, len(n.args))] += 1
            elif isinstance(block, LoopBlock):
                for t in block.body:
                    result[t.role] += 1
            else:
                # One executable binding template per formula role in a phase loop.
                for t in block.role_templates:
                    result[t.role] += 1
        return result


@dataclass
class _LoopAnalysis:
    by_id: dict[int, Any]
    schedule: tuple[int, ...]
    pos_of: dict[int, int]
    incoming: dict[int, tuple[int, ...]]

    @classmethod
    def from_program(cls, program: SequentialProgram) -> "_LoopAnalysis":
        schedule = tuple(op.node_id for op in program.ops)
        incoming_lists: dict[int, list[int]] = defaultdict(list)
        for src, dst in program.trace.dependencies:
            incoming_lists[dst].append(src)
        return cls(
            by_id=program.node_by_id,
            schedule=schedule,
            pos_of={nid: i for i, nid in enumerate(schedule)},
            incoming={nid: tuple(srcs) for nid, srcs in incoming_lists.items()},
        )


@dataclass(frozen=True)
class LoopCandidate:
    start: int
    body_size: int
    repetitions: int

    @property
    def stop(self) -> int:
        return self.start + self.body_size * self.repetitions

    @property
    def saved_ops(self) -> int:
        return self.body_size * (self.repetitions - 1)


@dataclass(frozen=True)
class _PhaseCandidate:
    role: tuple[str, str, int]
    positions: tuple[int, ...]
    variant_shapes: tuple[tuple[Any, ...], ...]
    phase_ids: tuple[int, ...]
    approximate_stored_ops: int

    @property
    def start(self) -> int:
        return self.positions[0]

    @property
    def stop(self) -> int:
        return self.positions[-1]

    @property
    def repetitions(self) -> int:
        return len(self.positions) - 1

    @property
    def phase_count(self) -> int:
        return len(self.variant_shapes)


def _candidate_periods(
    token_ids: Sequence[int],
    *,
    max_candidates: int,
    max_gap_neighbors: int,
    max_body_size: int | None,
) -> list[int]:
    positions: dict[int, list[int]] = defaultdict(list)
    for i, token in enumerate(token_ids):
        positions[token].append(i)

    counts: Counter[int] = Counter()
    for pos in positions.values():
        n = len(pos)
        for i in range(n):
            upper = min(n, i + 1 + max_gap_neighbors)
            for j in range(i + 1, upper):
                gap = pos[j] - pos[i]
                if gap <= 0:
                    continue
                if max_body_size is not None and gap > max_body_size:
                    continue
                counts[gap] += 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [gap for gap, _ in ranked[:max_candidates]]


def _runs_for_period(token_ids: Sequence[int], period: int, min_repetitions: int) -> list[tuple[int, int]]:
    n = len(token_ids)
    if period <= 0 or n < period * min_repetitions:
        return []
    limit = n - period
    runs = []
    i = 0
    minimum_true = period * (min_repetitions - 1)
    while i < limit:
        if token_ids[i] != token_ids[i + period]:
            i += 1
            continue
        start = i
        i += 1
        while i < limit and token_ids[i] == token_ids[i + period]:
            i += 1
        run_len = i - start
        if run_len >= minimum_true:
            reps = 1 + run_len // period
            if reps >= min_repetitions:
                runs.append((start, reps))
    return runs


def _dependency_regimes(
    analysis: _LoopAnalysis, start: int, body_size: int, repetitions: int
) -> tuple[tuple[int, ...], int]:
    """Describe realized dependency variants without splitting the code loop."""
    by_id = analysis.by_id
    schedule = analysis.schedule
    pos_of = analysis.pos_of
    incoming = analysis.incoming
    signatures = []
    for rep in range(repetitions):
        rep_sig = []
        for slot in range(body_size):
            pos = start + rep * body_size + slot
            nid = schedule[pos]
            deps = []
            for src in incoming.get(nid, ()):
                spos = pos_of.get(src)
                snode = by_id.get(src)
                if spos is not None and snode is not None:
                    rel = spos - start
                    src_rep = rel // body_size
                    src_slot = rel % body_size
                    if 0 <= src_slot < body_size:
                        template_node = by_id[schedule[start + src_slot]]
                        if snode.shape_token == template_node.shape_token:
                            deps.append(("relative", src_slot, src_rep - rep))
                            continue
                if snode is None:
                    deps.append(("external_unknown",))
                else:
                    deps.append(("external", snode.schema_uid, snode.space_family_uid, len(snode.args)))
            deps.sort()
            rep_sig.append(tuple(deps))
        signatures.append(tuple(rep_sig))

    ids: dict[Any, int] = {}
    variant_ids = []
    for sig in signatures:
        if sig not in ids:
            ids[sig] = len(ids)
        variant_ids.append(ids[sig])
    return tuple(variant_ids), len(ids)


def _make_loop_block(program: SequentialProgram, analysis: _LoopAnalysis, candidate: LoopCandidate) -> LoopBlock:
    by_id = analysis.by_id
    schedule = analysis.schedule
    body = []
    source_ids = tuple(schedule[candidate.start:candidate.stop])

    for slot in range(candidate.body_size):
        nodes = [
            by_id[schedule[candidate.start + rep * candidate.body_size + slot]]
            for rep in range(candidate.repetitions)
        ]
        first = nodes[0]
        if any(
            n.schema_uid != first.schema_uid
            or n.space_family_uid != first.space_family_uid
            or len(n.args) != len(first.args)
            for n in nodes[1:]
        ):
            raise LoopRecoveryError("candidate token match failed structural node validation")
        body.append(
            EvalTemplate(
                schema_uid=first.schema_uid,
                space_family_uid=first.space_family_uid,
                obj_binding=infer_binding([n.obj for n in nodes]),
                arg_bindings=tuple(infer_binding([n.args[j] for n in nodes]) for j in range(len(first.args))),
            )
        )

    regime_ids, regime_count = _dependency_regimes(
        analysis, candidate.start, candidate.body_size, candidate.repetitions
    )
    block = LoopBlock(
        start=candidate.start,
        body_size=candidate.body_size,
        repetitions=candidate.repetitions,
        body=tuple(body),
        dependency_variant_ids=regime_ids,
        dependency_variant_count=regime_count,
        source_node_ids=source_ids,
    )

    expected = tuple(by_id[nid].runtime_node for nid in source_ids)
    actual = block.expand_runtime_nodes()
    for i, (a, b) in enumerate(zip(actual, expected)):
        if a[0] is not b[0] or a[1] != b[1]:
            raise LoopRecoveryError(f"candidate binding reconstruction mismatch at {i}")
    return block


def _select_non_overlapping(candidates: Sequence[LoopCandidate]) -> list[LoopCandidate]:
    if not candidates:
        return []
    cands = sorted(candidates, key=lambda c: (c.stop, c.start, c.body_size))
    ends = [c.stop for c in cands]
    prev = [bisect.bisect_right(ends, c.start) - 1 for c in cands]
    score = [0] * (len(cands) + 1)
    for i, cand in enumerate(cands, 1):
        take = cand.saved_ops + score[prev[i - 1] + 1]
        score[i] = max(take, score[i - 1])

    selected = []
    i = len(cands)
    while i > 0:
        cand = cands[i - 1]
        take = cand.saved_ops + score[prev[i - 1] + 1]
        if take > score[i - 1]:
            selected.append(cand)
            i = prev[i - 1] + 1
        else:
            i -= 1
    selected.reverse()
    return selected


def _token_ids_for_range(analysis: _LoopAnalysis, start: int, stop: int) -> list[int]:
    ids: dict[Any, int] = {}
    out = []
    for nid in analysis.schedule[start:stop]:
        tok = analysis.by_id[nid].shape_token
        if tok not in ids:
            ids[tok] = len(ids)
        out.append(ids[tok])
    return out


def _recover_local_blocks(
    program: SequentialProgram,
    analysis: _LoopAnalysis,
    start: int,
    stop: int,
    *,
    min_repetitions: int,
    max_candidates: int,
    max_gap_neighbors: int,
    max_body_size: int | None,
) -> tuple[Block, ...]:
    if start >= stop:
        return ()
    token_ids = _token_ids_for_range(analysis, start, stop)
    periods = _candidate_periods(
        token_ids,
        max_candidates=max_candidates,
        max_gap_neighbors=max_gap_neighbors,
        max_body_size=max_body_size,
    )
    candidates: dict[tuple[int, int], LoopCandidate] = {}
    for period in periods:
        for local_start, reps in _runs_for_period(token_ids, period, min_repetitions):
            cand = LoopCandidate(start + local_start, period, reps)
            if cand.stop > stop:
                continue
            key = (cand.start, cand.stop)
            old = candidates.get(key)
            if old is None or cand.body_size < old.body_size:
                candidates[key] = cand

    selected = _select_non_overlapping(tuple(candidates.values()))
    blocks: list[Block] = []
    cursor = start
    for cand in selected:
        if cand.start > cursor:
            blocks.append(LiteralBlock(tuple(program.ops[cursor:cand.start])))
        blocks.append(_make_loop_block(program, analysis, cand))
        cursor = cand.stop
    if cursor < stop:
        blocks.append(LiteralBlock(tuple(program.ops[cursor:stop])))
    if not blocks:
        blocks.append(LiteralBlock(tuple(program.ops[start:stop])))
    return tuple(blocks)


def _phase_candidates(
    program: SequentialProgram,
    analysis: _LoopAnalysis,
    *,
    start: int = 0,
    stop: int | None = None,
    min_occurrences: int,
    max_count_clusters: int,
    max_phase_variants: int,
    min_coverage: float,
) -> list[_PhaseCandidate]:
    n_total = len(program.ops)
    if stop is None:
        stop = n_total
    if start < 0 or stop > n_total or start >= stop:
        return []
    n = stop - start

    role_positions: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    token_by_pos: list[tuple[str, str, int] | None] = [None] * n_total
    for pos in range(start, stop):
        node = analysis.by_id[analysis.schedule[pos]]
        role = (node.schema_uid, node.space_family_uid, len(node.args))
        role_positions[role].append(pos)
        token_by_pos[pos] = role

    count_hist = Counter(len(v) for v in role_positions.values() if len(v) >= min_occurrences)
    ranked_counts = sorted(count_hist, key=lambda c: (-(c * count_hist[c]), -count_hist[c], c))[
        :max_count_clusters
    ]

    result: list[_PhaseCandidate] = []
    for count in ranked_counts:
        for role, positions0 in role_positions.items():
            if len(positions0) != count:
                continue
            positions = tuple(positions0)
            covered = positions[-1] - positions[0]
            if covered <= 0 or covered / n < min_coverage:
                continue

            shape_to_id: dict[tuple[Any, ...], int] = {}
            variant_shapes: list[tuple[Any, ...]] = []
            phase_ids = []
            for i in range(len(positions) - 1):
                a, b = positions[i], positions[i + 1]
                shape = tuple(token_by_pos[a:b])
                if any(role is None for role in shape):
                    raise LoopRecoveryError("phase range tokenization gap")
                pid = shape_to_id.get(shape)
                if pid is None:
                    pid = len(variant_shapes)
                    if pid >= max_phase_variants:
                        phase_ids = []
                        break
                    shape_to_id[shape] = pid
                    variant_shapes.append(shape)
                phase_ids.append(pid)
            if not phase_ids:
                continue

            # Candidate generation must stay cheap on very large traces. Building
            # a RePair grammar for every possible anchor can repeat nearly the
            # same work dozens of times. Rank first using the unfactored exact
            # phase shapes, then grammar-score only a small generic shortlist.
            unique_roles = len({r for shape in variant_shapes for r in shape})
            raw_symbols = sum(len(shape) for shape in variant_shapes)
            approximate = (
                (positions[0] - start)
                + (stop - positions[-1])
                + unique_roles
                + raw_symbols
            )
            result.append(
                _PhaseCandidate(
                    role=role,
                    positions=positions,
                    variant_shapes=tuple(variant_shapes),
                    phase_ids=tuple(phase_ids),
                    approximate_stored_ops=approximate,
                )
            )

    result.sort(key=lambda c: (c.approximate_stored_ops, c.phase_count, -c.repetitions, c.start))
    shortlist = result[: min(8, len(result))]
    rescored: list[_PhaseCandidate] = []
    for cand in shortlist:
        grammar = _build_phase_grammar(cand.variant_shapes)
        exact_score = (
            (cand.positions[0] - start)
            + (stop - cand.positions[-1])
            + grammar.stored_symbols
            + grammar.terminal_count
        )
        rescored.append(
            _PhaseCandidate(
                role=cand.role,
                positions=cand.positions,
                variant_shapes=cand.variant_shapes,
                phase_ids=cand.phase_ids,
                approximate_stored_ops=exact_score,
            )
        )
    rescored.sort(key=lambda c: (c.approximate_stored_ops, c.phase_count, -c.repetitions, c.start))
    return rescored


def _build_phase_grammar(variant_shapes: Sequence[Sequence[tuple[str, str, int]]]) -> PhaseGrammar:
    terminal_id: dict[tuple[str, str, int], int] = {}
    terminal_roles: list[tuple[str, str, int]] = []
    programs: list[list[int]] = []
    for shape in variant_shapes:
        row = []
        for role in shape:
            tid = terminal_id.get(role)
            if tid is None:
                tid = len(terminal_roles)
                terminal_id[role] = tid
                terminal_roles.append(role)
            row.append(tid)
        programs.append(row)

    # RePair: repeatedly replace the most frequent adjacent pair. Ties are
    # deterministic by pair IDs. This is exact grammar factoring, not semantic
    # recurrence inference.
    rules: list[tuple[int, int]] = []
    next_symbol = len(terminal_roles)
    while True:
        counts: Counter[tuple[int, int]] = Counter()
        for row in programs:
            counts.update(zip(row, row[1:]))
        if not counts:
            break
        max_freq = max(counts.values())
        if max_freq < 2:
            break
        pair = min(pair for pair, freq in counts.items() if freq == max_freq)
        replacement = next_symbol
        next_symbol += 1
        replaced_total = 0
        new_programs: list[list[int]] = []
        for row in programs:
            out = []
            i = 0
            while i < len(row):
                if i + 1 < len(row) and (row[i], row[i + 1]) == pair:
                    out.append(replacement)
                    i += 2
                    replaced_total += 1
                else:
                    out.append(row[i])
                    i += 1
            new_programs.append(out)
        if replaced_total < 2:
            break
        rules.append(pair)
        programs = new_programs

    grammar = PhaseGrammar(
        terminal_roles=tuple(terminal_roles),
        rules=tuple(rules),
        variant_programs=tuple(tuple(row) for row in programs),
    )
    # Internal exactness check for the grammar itself.
    for pid, shape in enumerate(variant_shapes):
        expanded = grammar.expand_variant(pid)
        expected = tuple(terminal_id[role] for role in shape)
        if expanded != expected:
            raise LoopRecoveryError(f"phase grammar expansion mismatch for variant {pid}")
    return grammar


def _loop_block_variant_shapes(block: LoopBlock | PhaseLoopBlock) -> tuple[tuple[tuple[str, str, int], ...], ...]:
    if isinstance(block, LoopBlock):
        return (tuple(t.role for t in block.body),)
    result = []
    for pid in range(block.phase_count):
        result.append(tuple(block.grammar.terminal_roles[i] for i in block.grammar.expand_variant(pid)))
    return tuple(result)


def _recover_loop_families(blocks: Sequence[Block]) -> tuple[LoopFamilyTemplate, ...]:
    """Factor loop code across disjoint regions using exact role-set overlap.

    This does not merge execution state or reorder operations. A singleton loop
    can share only with another singleton of the identical role. Larger loops
    require at least three shared roles and 75% containment of the smaller role
    set, preventing generic one-cell helpers from gluing unrelated loops together.
    """
    records = []
    for idx, block in enumerate(blocks):
        if not isinstance(block, (LoopBlock, PhaseLoopBlock)):
            continue
        shapes = _loop_block_variant_shapes(block)
        roles = frozenset(r for shape in shapes for r in shape)
        records.append((idx, shapes, roles))
    records.sort(key=lambda x: (-len(x[2]), x[0]))

    groups: list[dict[str, Any]] = []
    for idx, shapes, roles in records:
        chosen = None
        for group in groups:
            rep = group["rep"]
            shared = len(roles & rep)
            if len(roles) == len(rep) == 1:
                compatible = roles == rep
            else:
                compatible = shared >= 3 and shared / min(len(roles), len(rep)) >= 0.75
            if compatible:
                chosen = group
                break
        if chosen is None:
            chosen = {"rep": roles, "members": [], "shapes": []}
            groups.append(chosen)
        chosen["members"].append(idx)
        chosen["shapes"].extend(shapes)

    families = []
    for group in groups:
        # Identical variants from several instances need only one code variant.
        unique_shapes = tuple(dict.fromkeys(group["shapes"]))
        grammar = _build_phase_grammar(unique_shapes)
        families.append(
            LoopFamilyTemplate(
                member_block_indices=tuple(sorted(group["members"])),
                grammar=grammar,
            )
        )
    families.sort(key=lambda f: f.member_block_indices[0])
    return tuple(families)


def _formula_registry(program: SequentialProgram) -> tuple[tuple[FormulaOp, ...], dict[tuple[str, str, int], int]]:
    """Intern every structural formula role once in first-realized-use order."""
    by_id = program.node_by_id
    role_to_id: dict[tuple[str, str, int], int] = {}
    ops: list[FormulaOp] = []
    for eval_op in program.ops:
        node = by_id[eval_op.node_id]
        role = node.shape_token
        if role in role_to_id:
            continue
        op_id = len(ops)
        role_to_id[role] = op_id
        ops.append(
            FormulaOp(
                op_id=op_id,
                schema_uid=node.schema_uid,
                space_family_uid=node.space_family_uid,
                arity=len(node.args),
                name=node.schema.name,
                fullname=node.schema.fullname,
            )
        )
    return tuple(ops), role_to_id


def _role_bindings_for_block(
    block: LoopBlock | PhaseLoopBlock,
    role_to_op_id: dict[tuple[str, str, int], int],
) -> dict[tuple[str, str, int], RoleBindingTemplate]:
    """Collapse all callsites of one role in one loop instance into one binding stream."""
    if isinstance(block, PhaseLoopBlock):
        return {
            t.role: RoleBindingTemplate(
                formula_op_id=role_to_op_id[t.role],
                obj_binding=t.obj_binding,
                arg_bindings=t.arg_bindings,
            )
            for t in block.role_templates
        }

    roles = tuple(dict.fromkeys(t.role for t in block.body))
    if all(sum(t.role == role for t in block.body) == 1 for role in roles):
        # Common fast path: one body slot per formula role. Existing bindings are
        # already ordered by loop occurrence and need no materialization.
        return {
            t.role: RoleBindingTemplate(
                formula_op_id=role_to_op_id[t.role],
                obj_binding=t.obj_binding,
                arg_bindings=t.arg_bindings,
            )
            for t in block.body
        }

    objects: dict[tuple[str, str, int], list[Any]] = {r: [] for r in roles}
    args: dict[tuple[str, str, int], list[list[Any]]] = {
        r: [[] for _ in range(r[2])] for r in roles
    }
    for i in range(block.repetitions):
        for tmpl in block.body:
            role = tmpl.role
            obj, key = tmpl.runtime_node(i)
            objects[role].append(obj)
            for j, value in enumerate(key):
                args[role][j].append(value)

    result = {}
    for role in roles:
        result[role] = RoleBindingTemplate(
            formula_op_id=role_to_op_id[role],
            obj_binding=infer_binding(objects[role]),
            arg_bindings=tuple(infer_binding(v) for v in args[role]),
        )
    return result


def _build_canonical_execution_plan(
    program: SequentialProgram,
    blocks: Sequence[Block],
    loop_families: Sequence[LoopFamilyTemplate],
) -> CanonicalExecutionPlan:
    """Make loop-family sharing executable instead of merely diagnostic."""
    formula_ops, role_to_op_id = _formula_registry(program)

    code_families: list[LoopCodeFamily] = []
    block_family: dict[int, int] = {}
    for family_id, family in enumerate(loop_families):
        code_families.append(
            LoopCodeFamily(
                family_id=family_id,
                formula_op_ids=tuple(role_to_op_id[r] for r in family.grammar.terminal_roles),
                grammar=family.grammar,
            )
        )
        for block_index in family.member_block_indices:
            if block_index in block_family:
                raise LoopRecoveryError("loop block assigned to more than one code family")
            block_family[block_index] = family_id

    instances: list[LoopInstancePlan] = []
    block_to_instance: list[int | None] = [None] * len(blocks)
    for block_index, block in enumerate(blocks):
        if not isinstance(block, (LoopBlock, PhaseLoopBlock)):
            continue
        if block_index not in block_family:
            raise LoopRecoveryError("loop block has no shared code family")
        family_id = block_family[block_index]
        family = loop_families[family_id]
        code_family = code_families[family_id]

        family_shapes = []
        for pid in range(len(family.grammar.variant_programs)):
            family_shapes.append(
                tuple(family.grammar.terminal_roles[i] for i in family.grammar.expand_variant(pid))
            )
        shape_to_variant = {shape: i for i, shape in enumerate(family_shapes)}
        block_shapes = _loop_block_variant_shapes(block)
        try:
            local_to_family = tuple(shape_to_variant[shape] for shape in block_shapes)
        except KeyError as exc:
            raise LoopRecoveryError("loop-family grammar does not contain member variant") from exc

        if isinstance(block, LoopBlock):
            variant_ids = (local_to_family[0],) * block.repetitions
        else:
            variant_ids = tuple(local_to_family[pid] for pid in block.phase_ids)

        by_role = _role_bindings_for_block(block, role_to_op_id)
        aligned: list[RoleBindingTemplate | None] = []
        for role, op_id in zip(family.grammar.terminal_roles, code_family.formula_op_ids):
            binding = by_role.get(role)
            if binding is not None and binding.formula_op_id != op_id:
                raise LoopRecoveryError("formula-op registry mismatch")
            aligned.append(binding)

        instance_index = len(instances)
        block_to_instance[block_index] = instance_index
        instances.append(
            LoopInstancePlan(
                block_index=block_index,
                family_id=family_id,
                variant_ids=variant_ids,
                role_bindings=tuple(aligned),
            )
        )

    return CanonicalExecutionPlan(
        formula_ops=formula_ops,
        code_families=tuple(code_families),
        loop_instances=tuple(instances),
        block_to_instance=tuple(block_to_instance),
    )


def _make_phase_loop(program: SequentialProgram, analysis: _LoopAnalysis, cand: _PhaseCandidate) -> PhaseLoopBlock:
    by_id = analysis.by_id
    schedule = analysis.schedule
    grammar = _build_phase_grammar(cand.variant_shapes)

    # One exact binding stream per formula role over the whole phase region.
    nodes_by_role: dict[tuple[str, str, int], list[Any]] = defaultdict(list)
    for nid in schedule[cand.start:cand.stop]:
        node = by_id[nid]
        role = (node.schema_uid, node.space_family_uid, len(node.args))
        nodes_by_role[role].append(node)

    role_templates: list[EvalTemplate] = []
    for role in grammar.terminal_roles:
        nodes = nodes_by_role[role]
        if not nodes:
            raise LoopRecoveryError("phase grammar terminal has no concrete nodes")
        first = nodes[0]
        role_templates.append(
            EvalTemplate(
                schema_uid=first.schema_uid,
                space_family_uid=first.space_family_uid,
                obj_binding=infer_binding([n.obj for n in nodes]),
                arg_bindings=tuple(
                    infer_binding([n.args[j] for n in nodes]) for j in range(len(first.args))
                ),
            )
        )

    block = PhaseLoopBlock(
        start=cand.start,
        stop=cand.stop,
        anchor_role=cand.role,
        phase_ids=cand.phase_ids,
        grammar=grammar,
        role_templates=tuple(role_templates),
    )

    expected = tuple(by_id[nid].runtime_node for nid in schedule[cand.start:cand.stop])
    actual = block.expand_runtime_nodes()
    if len(expected) != len(actual):
        raise LoopRecoveryError("phase loop expansion length mismatch")
    for i, (a, b) in enumerate(zip(actual, expected)):
        if a[0] is not b[0] or a[1] != b[1]:
            raise LoopRecoveryError(f"phase binding reconstruction mismatch at {i}")
    return block


def _stored_role_counts_for_blocks(
    program: SequentialProgram, blocks: Sequence[Block]
) -> Counter[tuple[str, str, int]]:
    temp = StructuredSequentialProgram(program, tuple(blocks), recovery_mode="analysis")
    return temp.stored_role_counts()


def _recover_range(
    program: SequentialProgram,
    analysis: _LoopAnalysis,
    start: int,
    stop: int,
    *,
    min_repetitions: int,
    max_candidates: int,
    max_gap_neighbors: int,
    max_body_size: int | None,
    enable_phase_loops: bool,
    phase_min_occurrences: int,
    phase_count_clusters: int,
    max_phase_variants: int,
    phase_min_coverage: float,
    phase_complexity_cap: float,
    max_phase_depth: int,
    depth: int,
) -> tuple[Block, ...]:
    local = _recover_local_blocks(
        program,
        analysis,
        start,
        stop,
        min_repetitions=min_repetitions,
        max_candidates=max_candidates,
        max_gap_neighbors=max_gap_neighbors,
        max_body_size=max_body_size,
    )
    if (
        not enable_phase_loops
        or depth >= max_phase_depth
        or stop - start < max(16, phase_min_occurrences * 2)
    ):
        return local

    concrete_counts: Counter[tuple[str, str, int]] = Counter()
    for nid in analysis.schedule[start:stop]:
        n = analysis.by_id[nid]
        concrete_counts[(n.schema_uid, n.space_family_uid, len(n.args))] += 1
    recurrent = {role for role, count in concrete_counts.items() if count >= 3}

    def block_quality(blocks: Sequence[Block]) -> tuple[int, int, int, int, int]:
        counts = _stored_role_counts_for_blocks(program, blocks)
        bad = sum(counts[role] > 2 for role in recurrent)
        excess = sum(max(0, counts[role] - 2) for role in recurrent)
        max_count = max((counts[role] for role in recurrent), default=0)
        stored = sum(b.stored_body_ops for b in blocks)
        control = sum(len(b.phase_ids) for b in blocks if isinstance(b, PhaseLoopBlock))
        return (bad, excess, max_count, stored, control)

    best = local
    best_quality = block_quality(best)
    baseline_storage = sum(b.stored_body_ops for b in local)

    candidates = _phase_candidates(
        program,
        analysis,
        start=start,
        stop=stop,
        min_occurrences=phase_min_occurrences,
        max_count_clusters=phase_count_clusters,
        max_phase_variants=max_phase_variants,
        min_coverage=phase_min_coverage,
    )
    if not candidates:
        return best

    # Only the strongest candidate of the dominant recurrence-count cluster is
    # considered at each range. Prefix/suffix recursion then discovers independent
    # axes in disjoint regions without allowing a low-support nested helper to
    # replace the dominant outer recurrence.
    cand = candidates[0]
    phase = _make_phase_loop(program, analysis, cand)
    prefix = _recover_range(
        program, analysis, start, cand.start,
        min_repetitions=min_repetitions,
        max_candidates=max_candidates,
        max_gap_neighbors=max_gap_neighbors,
        max_body_size=max_body_size,
        enable_phase_loops=enable_phase_loops,
        phase_min_occurrences=phase_min_occurrences,
        phase_count_clusters=phase_count_clusters,
        max_phase_variants=max_phase_variants,
        phase_min_coverage=phase_min_coverage,
        phase_complexity_cap=phase_complexity_cap,
        max_phase_depth=max_phase_depth,
        depth=depth + 1,
    )
    suffix = _recover_range(
        program, analysis, cand.stop, stop,
        min_repetitions=min_repetitions,
        max_candidates=max_candidates,
        max_gap_neighbors=max_gap_neighbors,
        max_body_size=max_body_size,
        enable_phase_loops=enable_phase_loops,
        phase_min_occurrences=phase_min_occurrences,
        phase_count_clusters=phase_count_clusters,
        max_phase_variants=max_phase_variants,
        phase_min_coverage=phase_min_coverage,
        phase_complexity_cap=phase_complexity_cap,
        max_phase_depth=max_phase_depth,
        depth=depth + 1,
    )
    trial = prefix + (phase,) + suffix
    trial_quality = block_quality(trial)
    trial_storage = sum(b.stored_body_ops for b in trial)
    cap = max(baseline_storage + 64, int(np.ceil(baseline_storage * phase_complexity_cap)))
    if trial_storage < baseline_storage or (trial_storage <= cap and trial_quality < best_quality):
        return trial
    return best


def recover_loops(
    program: SequentialProgram,
    *,
    min_repetitions: int = 3,
    max_candidates: int = 64,
    max_gap_neighbors: int = 8,
    max_body_size: int | None = 4096,
    enable_phase_loops: bool = True,
    phase_min_occurrences: int = 4,
    phase_count_clusters: int = 1,
    max_phase_variants: int = 64,
    phase_min_coverage: float = 0.10,
    phase_complexity_cap: float = 2.00,
    max_phase_depth: int = 4,
) -> StructuredSequentialProgram:
    """Recover a hierarchical exact loop/grammar representation of one trace.

    No model/product/Cell name or parameter name/position is interpreted. Local
    loops compress repeated contiguous role sequences. A phase loop is inferred
    from the recurrence-count cluster supported by the most formula roles in the
    current range. Exact observed phase IDs and global per-role binding streams
    retain branch-dependent topology and irregular arguments. Prefix/suffix ranges
    recurse independently, allowing unrelated preprocessing/projection axes to be
    recovered without confusing them with nested helper recurrences.
    """
    if min_repetitions < 2:
        raise ValueError("min_repetitions must be at least 2")
    if phase_complexity_cap < 1.0:
        raise ValueError("phase_complexity_cap must be at least 1.0")

    analysis = _LoopAnalysis.from_program(program)
    blocks = _recover_range(
        program,
        analysis,
        0,
        len(program.ops),
        min_repetitions=min_repetitions,
        max_candidates=max_candidates,
        max_gap_neighbors=max_gap_neighbors,
        max_body_size=max_body_size,
        enable_phase_loops=enable_phase_loops,
        phase_min_occurrences=phase_min_occurrences,
        phase_count_clusters=phase_count_clusters,
        max_phase_variants=max_phase_variants,
        phase_min_coverage=phase_min_coverage,
        phase_complexity_cap=phase_complexity_cap,
        max_phase_depth=max_phase_depth,
        depth=0,
    )
    mode = "phase" if any(isinstance(b, PhaseLoopBlock) for b in blocks) else "local"
    structured = StructuredSequentialProgram(program, tuple(blocks), recovery_mode=mode)
    structured.loop_families = _recover_loop_families(structured.blocks)
    structured.canonical_plan = _build_canonical_execution_plan(
        program, structured.blocks, structured.loop_families
    )
    structured.validate_exact_expansion()
    return structured
