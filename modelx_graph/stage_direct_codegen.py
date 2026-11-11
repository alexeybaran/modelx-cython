from __future__ import annotations

"""Direct source emission from :class:`StageOptimizedProgram`.

No ExecutableGraph, OptimizedProgram, legacy storage layout or coord_phase is
consulted.  Multi-stage programs execute the already-proven Stage order,
StageStoragePlan materialization contract, and admitted completion barriers
directly.  Physical fence admission is driven by the frozen StageStorageFence
execution mode, not semantic availability.  Cython intentionally remains GIL-bound while semantic/backend ownership
is being completed.
"""

import ast
import copy
from typing import Any

from .codegen_common import CodegenError
from .stage_optimized_program import StageOptimizedProgram


class StageDirectSourceBuilder:
    def __init__(self, program: StageOptimizedProgram):
        self.program = program
        if not program.direct_python_supported:
            raise CodegenError(
                "StageOptimizedProgram lacks a complete direct contract: "
                + "; ".join(program.direct_python_blockers)
            )
        if not program.execution_blocks:
            raise CodegenError("direct emitter requires at least one execution block")
        self.domains = {int(x.stage_index): x for x in program.iteration_domains}
        self.blocks = tuple(program.execution_blocks)
        post_stages = {
            int(x.stage_index) for x in self.blocks if x.kind == "post_stage_scalar"
        }
        coordinate_stages = {
            int(x.stage_index) for x in self.blocks if x.kind == "coordinate_scan"
        }
        unknown_kinds = sorted({x.kind for x in self.blocks} - {"coordinate_scan", "post_stage_scalar"})
        if unknown_kinds:
            raise CodegenError(f"unsupported direct execution block kinds: {unknown_kinds!r}")
        if post_stages & set(self.domains):
            raise CodegenError(
                f"post-stage scalar blocks must not own iteration domains: {tuple(sorted(post_stages & set(self.domains)))!r}"
            )
        if coordinate_stages != set(self.domains):
            raise CodegenError(
                f"coordinate execution blocks and iteration-domain stages disagree: "
                f"blocks={tuple(sorted(coordinate_stages))!r}, domains={tuple(sorted(self.domains))!r}"
            )
        covered_stages = set(self.domains) | post_stages
        if covered_stages != set(program.stage_indices):
            raise CodegenError(
                f"direct execution blocks do not cover Stage order {program.stage_indices!r}: "
                f"{tuple(sorted(covered_stages))!r}"
            )
        self.v = program.variants
        self.nodes = {x.uid: x for x in program.graph.nodes}
        self.physical = {x.value_uid: x for x in program.physical_values}
        self.inputs = {x.key: x for x in program.inputs}
        self.pure = {x.uid for x in program.pure_maps}
        self.aux = {x.uid: x for x in program.auxiliary_recurrences}
        self.scalar_slots = 1 + max(
            (int(x.working_scalar_slot) for x in program.physical_values if x.working_scalar_slot is not None),
            default=-1,
        )
        self.current_slots = 1 + max(
            (int(x.current_slot) for x in program.physical_values if x.current_slot is not None), default=-1
        )
        self.ring_slots = max(
            (int(x.ring_base) + int(x.ring_depth or 0) for x in program.physical_values if x.ring_base is not None),
            default=0,
        )
        self.history_slots = 1 + max(
            (int(x.history_slot) for x in program.physical_values if x.history_slot is not None), default=-1
        )
        self.materialized_slots = 1 + max(
            (int(x.materialized_slot) for x in program.physical_values if x.materialized_slot is not None), default=-1
        )
        self.carried_scalar_slots = 1 + max(
            (int(x.scalar_slot) for x in program.physical_values if x.scalar_slot is not None), default=-1
        )
        self.roles = {uid: getattr(cv, "role", "scalar") for uid, cv in self.v.items()}
        self.state_semantics = {x.uid: x.state_semantic for x in program.graph.nodes}
        self.execution_semantics = {x.uid: x.execution_semantic for x in program.graph.nodes}

    def _impl_source(self, uid: str) -> str:
        cv = self.v[uid]
        if getattr(cv, "role", None) in {"reduction", "vector"} or cv.function is None:
            return ""
        fn = copy.deepcopy(cv.function)
        fn.name = f"_impl_{uid}"
        ast.fix_missing_locations(fn)
        return ast.unparse(fn)

    @staticmethod
    def _tuple_source(parts: tuple[str, ...]) -> str:
        if not parts: return "()"
        if len(parts) == 1: return f"({parts[0]},)"
        return "(" + ", ".join(parts) + ")"

    def _reduction_helpers(self, uid: str) -> list[str]:
        spec = self.v[uid].reduction
        if spec is None:
            raise CodegenError(f"reduction {uid} lacks ReductionSpec")
        lines = [f"def _red_init_{uid}():", f"    return {ast.unparse(spec.init)}", ""]
        args = f"{spec.loop_var}, {spec.target}"
        lines += [f"def _red_filter_{uid}({args}):"]
        if spec.filters:
            lines.append("    return bool(" + " and ".join(f"({ast.unparse(x)})" for x in spec.filters) + ")")
        else:
            lines.append("    return True")
        lines += ["", f"def _red_body_{uid}({args}):", f"    return {ast.unparse(spec.body_expr)}", ""]
        return lines

    def _physical_literal(self) -> str:
        rows: dict[str, dict[str, Any]] = {}
        for uid, row in sorted(self.physical.items()):
            node = self.nodes[uid]
            rows[uid] = {
                "kind": row.working_kind,
                "current": row.current_slot,
                "ring_base": row.ring_base,
                "ring_depth": row.ring_depth,
                "history": row.history_slot,
                "scalar": row.working_scalar_slot,
                "materialized": row.materialized_slot,
                "carried": row.scalar_slot,
                "producer_stage": row.producer_stage_index,
                "consumers": tuple(row.consumer_stage_indices),
                "retained_offset_min": row.retained_offset_min,
                "retained_offset_max": row.retained_offset_max,
                "retained_bounds": row.retained_bounds,
                "domain": node.domain,
            }
        return repr(rows)

    def _guard_lines(self) -> list[str]:
        lines = ["def __guard_fail__(code):"]
        if not self.program.guards:
            return lines + ["    raise RuntimeError(f'unknown optimized guard code {code}')", ""]
        for i, guard in enumerate(self.program.guards):
            lines.append(f"    {'if' if i == 0 else 'elif'} code == {int(guard.code)}:")
            if guard.message is None:
                lines.append(f"        raise {guard.exception_type}")
            else:
                lines.append(f"        raise {guard.exception_type}({guard.message!r})")
        lines += ["    else:", "        raise RuntimeError(f'unknown optimized guard code {code}')", ""]
        return lines

    def _core_lines(self) -> list[str]:
        p = self.program
        stage_order = tuple(int(x) for x in p.stage_indices)
        block_rows = tuple(
            (
                int(x.stage_index), x.iteration_domain_uid,
                x.persistent_driver_uids, x.reduction_uids, x.scan_direction,
            )
            for x in self.blocks if x.kind == "coordinate_scan"
        )
        post_stage_scalars: dict[int, tuple[str, ...]] = {}
        for stage in stage_order:
            post_stage_scalars[stage] = tuple(sorted({
                uid
                for block in self.blocks
                if block.stage_index == stage and block.kind == "post_stage_scalar"
                for uid in (*block.pre_scalar_uids, *block.post_scalar_uids)
            }))
        reductions = tuple(uid for block in self.blocks for uid in block.reduction_uids)
        domain_rows = tuple(
            (
                int(d.stage_index), d.uid, tuple(d.range_arg_sources),
                int(d.coordinate_step), d.scan_direction,
                int(d.materialization_extension_min), int(d.materialization_extension_max),
            )
            for d in sorted(self.domains.values(), key=lambda x: int(x.stage_index))
        )
        carried_by_stage: dict[int, tuple[str, ...]] = {}
        materialized_by_stage: dict[int, tuple[str, ...]] = {}
        for stage in stage_order:
            carried_by_stage[stage] = tuple(sorted(
                uid for uid, row in self.physical.items()
                if row.scalar_slot is not None and row.producer_stage_index == stage
            ))
            materialized_by_stage[stage] = tuple(sorted(
                uid for uid, row in self.physical.items()
                if row.materialized_slot is not None and row.producer_stage_index == stage
            ))
        barrier_rows = tuple(
            (
                x.uid,
                int(x.source_stage_index) if x.source_stage_index is not None else None,
                int(x.target_stage_index) if x.target_stage_index is not None else None,
                x.availability,
                x.execution_mode,
                tuple(x.required_materialized_value_uids),
                tuple(int(slot) for slot in x.required_materialized_slots),
                x.proof_kind,
                tuple(x.evidence_uids),
                x.execution_mode_proof_kind,
            )
            for x in p.barriers
        )
        incoming_barriers: dict[int, tuple[str, ...]] = {
            int(stage): tuple(x.uid for x in p.barriers if x.target_stage_index == stage)
            for stage in stage_order
        }
        outgoing_barriers: dict[int, tuple[str, ...]] = {
            int(stage): tuple(x.uid for x in p.barriers if x.source_stage_index == stage)
            for stage in stage_order
        }
        lines: list[str] = [
            "import math", "import numpy as np", "",
            f"STAGE_PROGRAM_UID = {p.uid!r}",
            f"STAGE_EXECUTION_PLAN_UID = {p.execution_plan.uid!r}",
            f"STAGE_STORAGE_PLAN_UID = {p.storage_plan.uid!r}",
            f"STAGE_RUN_DOMAIN_UID = {p.run_domain.uid!r}",
            f"STAGE_RUN_POINT_COUNT = {int(p.run_domain.point_count)}",
            f"FORMULA_HASH = {p.graph.formula_hash!r}",
            f"_PHYSICAL = {self._physical_literal()}",
            f"_ROLES = {self.roles!r}",
            f"_STATE = {self.state_semantics!r}",
            f"_EXECUTION = {self.execution_semantics!r}",
            f"_PURE_MAPS = {frozenset(self.pure)!r}",
            f"_AUXILIARY = {frozenset(self.aux)!r}",
            f"_STAGE_ORDER = {stage_order!r}",
            f"_STAGE_DOMAINS = {domain_rows!r}",
            f"_BLOCKS = {block_rows!r}",
            f"_POST_STAGE_SCALARS = {post_stage_scalars!r}",
            f"_CARRIED_BY_STAGE = {carried_by_stage!r}",
            f"_MATERIALIZED_BY_STAGE = {materialized_by_stage!r}",
            f"_BARRIERS = {barrier_rows!r}",
            f"_INCOMING_BARRIERS = {incoming_barriers!r}",
            f"_OUTGOING_BARRIERS = {outgoing_barriers!r}",
            f"_REDUCTIONS = {reductions!r}",
            f"_BOUNDARY_SEEDS = {tuple((int(x.stage_index), x.value_uid, int(x.coordinate), tuple(int(q) for q in x.active_boundary_coordinates), None if x.replay_start_coordinate is None else int(x.replay_start_coordinate)) for x in p.boundary_seeds)!r}",
            f"_SCALAR_SLOT_COUNT = {int(self.scalar_slots)}",
            f"_CURRENT_SLOT_COUNT = {int(self.current_slots)}",
            f"_RING_SLOT_COUNT = {int(self.ring_slots)}",
            f"_HISTORY_SLOT_COUNT = {int(self.history_slots)}",
            f"_MATERIALIZED_SLOT_COUNT = {int(self.materialized_slots)}",
            f"_CARRIED_SCALAR_SLOT_COUNT = {int(self.carried_scalar_slots)}", "",
        ]
        lines += self._guard_lines()
        lines += [
            "_ACTIVE = None", "",
            "def point_input(key):", "    return _ACTIVE.point_input(key)", "",
            "def global_input(key):", "    return _ACTIVE.global_input(key)", "",
            "def array_input(key, index):", "    return _ACTIVE.array_input(key, index)", "",
            "def table_input(key, *indices):", "    return _ACTIVE.table_input(key, *indices)", "",
            "def __exact_axis_code__(value, start, step, size):",
            "    x = int(value)",
            "    if x != value:", "        raise KeyError(value)",
            "    delta = x - int(start)",
            "    if int(step) == 0 or delta % int(step) != 0:", "        raise KeyError(value)",
            "    idx = delta // int(step)",
            "    if idx < 0 or idx >= int(size):", "        raise KeyError(value)",
            "    return idx", "",
            "def __sparse_table_lookup__(keys_key, values_key, *indices):",
            "    return _ACTIVE.sparse_table_input(keys_key, values_key, *indices)", "",
            "def __prepared_point_lookup__(value_key, valid_key):",
            "    if not bool(_ACTIVE.point_input(valid_key)):", "        raise KeyError(('prepared_point_lookup', value_key))",
            "    return _ACTIVE.point_input(value_key)", "",
            "def __is_missing__(value):", "    return math.isnan(float(value))", "",
            "def __is_inf__(value):", "    return math.isinf(float(value))", "",
            "def __erf__(value):", "    return math.erf(float(value))", "",
            "def __step_lookup_1d__(axis_key, values_key, q, default, below_mode):",
            "    axis = _ACTIVE.inputs[axis_key]", "    values = _ACTIVE.inputs[values_key]",
            "    j = int(np.searchsorted(axis, q, side='right')) - 1",
            "    if j < 0:", "        return float(values[0]) if int(below_mode) == 1 else float(default)",
            "    return float(values[j])", "",
            "def __interval_lookup_1d__(lower_key, upper_key, values_key, q, closure_mode=0, fallback_mode=0, default=0.0):",
            "    lower = _ACTIVE.inputs[lower_key]", "    upper = _ACTIVE.inputs[upper_key]", "    values = _ACTIVE.inputs[values_key]",
            "    for j in range(len(values)):",
            "        if (int(closure_mode) == 1 and lower[j] <= q <= upper[j]) or (int(closure_mode) == 0 and lower[j] < q <= upper[j]):",
            "            return float(values[j])",
            "    if int(fallback_mode) == 0 or q > upper[-1]:", "        return float(values[-1])",
            "    return float(default)", "",
            "def __interp_lookup_1d__(axis_key, values_key, q, mode):",
            "    axis = _ACTIVE.inputs[axis_key]", "    values = _ACTIVE.inputs[values_key]", "    n = len(axis)",
            "    if q <= axis[0]:", "        lo = 0", "    elif q >= axis[n - 1]:", "        lo = n - 2",
            "    else:", "        lo = int(np.searchsorted(axis, q, side='right')) - 1",
            "    hi = lo + 1", "    frac = (q - axis[lo]) / (axis[hi] - axis[lo])",
            "    if int(mode) == 1:", "        return float(values[lo] * (values[hi] / values[lo]) ** frac)",
            "    return float(values[lo] + (values[hi] - values[lo]) * frac)", "",
        ]
        for uid in sorted(self.v):
            src = self._impl_source(uid)
            if src:
                lines += [src, ""]
        for uid in reductions:
            lines += self._reduction_helpers(uid)
        for uid in sorted(self.v):
            lines += [f"def {uid}(*args):", f"    return _ACTIVE.call({uid!r}, args)", ""]
        for stage, _uid, sources, _step, _scan, _ext_min, _ext_max in domain_rows:
            range_tuple = self._tuple_source(tuple(sources))
            lines += [
                f"def _direct_range_args_stage_{stage}():",
                f"    return {range_tuple}", "",
            ]
        lines += ["_IMPLS = {"]
        for uid in sorted(self.v):
            if self._impl_source(uid):
                lines.append(f"    {uid!r}: _impl_{uid},")
        lines += ["}", ""]
        lines += [
            "class _StageKernel:",
            "    def __init__(self, point_index, inputs):",
            "        self.i = int(point_index)", "        self.inputs = inputs",
            "        self.svals = [0.0] * max(1, _SCALAR_SLOT_COUNT)", "        self.scalar_ready = set()",
            "        self.carried = [0.0] * max(1, _CARRIED_SCALAR_SLOT_COUNT)",
            "        self.carried_valid = [False] * max(1, _CARRIED_SCALAR_SLOT_COUNT)",
            "        self.current = [0.0] * max(1, _CURRENT_SLOT_COUNT)", "        self.current_coord = [None] * max(1, _CURRENT_SLOT_COUNT)",
            "        self.ring = [0.0] * max(1, _RING_SLOT_COUNT)", "        self.ring_coord = [None] * max(1, _RING_SLOT_COUNT)",
            "        self.history = None", "        self.history_valid = None",
            "        self.materialized = [None] * max(1, _MATERIALIZED_SLOT_COUNT)",
            "        self.materialized_valid = [None] * max(1, _MATERIALIZED_SLOT_COUNT)",
            "        self.sparse_lookup_cache = {}",
            "        self.seeds = {}",
            "        self.seed_specs = {(stage, uid, q): (active, replay) for stage, uid, q, active, replay in _BOUNDARY_SEEDS}",
            "        self.replay_values = {}",
            "        self.coordinate_cache = {}", "        self.active = set()", "        self.current_q = None",
            "        self.current_stage = None", "        self.stage_geometry = {}",
            "        self.satisfied_barriers = set()", "",
            "    def point_input(self, key):",
            "        value = self.inputs[key]", "        arr = np.asarray(value)",
            "        if arr.ndim == 0:", "            return value", "        return value[self.i]", "",
            "    def global_input(self, key):", "        return self.inputs[key]", "",
            "    def array_input(self, key, index):", "        return self.inputs[key][int(index)]", "",
            "    def table_input(self, key, *indices):", "        return self.inputs[key][tuple(int(x) for x in indices)]", "",
            "    def sparse_table_input(self, keys_key, values_key, *indices):",
            "        cache_key = (keys_key, values_key)",
            "        mapping = self.sparse_lookup_cache.get(cache_key)",
            "        if mapping is None:",
            "            keys = np.asarray(self.inputs[keys_key])", "            values = np.asarray(self.inputs[values_key])",
            "            mapping = {tuple(int(x) for x in keys[j]): values[j] for j in range(len(values))}",
            "            self.sparse_lookup_cache[cache_key] = mapping",
            "        key = tuple(int(x) for x in indices)",
            "        if key not in mapping:", "            raise KeyError(key)",
            "        return mapping[key]", "",
            "    def _invoke(self, uid, args):",
            "        try:", "            fn = _IMPLS[uid]", "        except KeyError as exc:",
            "            raise RuntimeError(f'canonical value {uid} has no executable function') from exc",
            "        return fn(*args)", "",
            "    def call(self, uid, args):",
            "        role = _ROLES[uid]",
            "        if role == 'scalar':",
            "            if args:", "                raise RuntimeError(f'scalar canonical value {uid} received arguments')",
            "            return self.scalar(uid)",
            "        if role == 'reduction':",
            "            if args:", "                raise RuntimeError(f'reduction canonical value {uid} received arguments')",
            "            return self.reduction(uid)",
            "        if role == 'vector':", "            raise RuntimeError(f'vector canonical value outside direct subset: {uid}')",
            "        if len(args) != 1:", "            raise RuntimeError(f'coordinate canonical value {uid} requires one coordinate')",
            "        q = int(args[0])",
            "        if uid in _PURE_MAPS:", "            return self._invoke(uid, (q,))",
            "        if uid in _AUXILIARY:", "            return self.auxiliary(uid, q)",
            "        if _STATE[uid] == 'persistent_state':", "            return self.persistent(uid, q)",
            "        key = (uid, q)", "        if key in self.coordinate_cache:", "            return self.coordinate_cache[key]",
            "        active_key = ('coord', uid, q)", "        if active_key in self.active:",
            "            raise RuntimeError(f'nonpersistent coordinate recursion cycle: {uid}({q})')",
            "        self.active.add(active_key)", "        try:", "            value = self._invoke(uid, (q,))",
            "            self.coordinate_cache[key] = value", "            return value", "        finally:", "            self.active.remove(active_key)", "",
            "    def scalar(self, uid):",
            "        row = _PHYSICAL[uid]", "        carried = row['carried']", "        producer = row['producer_stage']",
            "        if carried is not None and self.current_stage is not None and producer is not None and self.current_stage > producer:",
            "            if not self.carried_valid[carried]:",
            "                raise RuntimeError(f'stage-carried scalar {uid} unavailable in stage {self.current_stage}')",
            "            return self.carried[carried]",
            "        slot = row['scalar']",
            "        if slot is None:", "            raise RuntimeError(f'scalar {uid} lacks Stage physical scalar slot')",
            "        if uid in self.scalar_ready:", "            return self.svals[slot]",
            "        active_key = ('scalar', uid)", "        if active_key in self.active:", "            raise RuntimeError(f'scalar dependency cycle: {uid}')",
            "        self.active.add(active_key)", "        try:", "            value = self._invoke(uid, ())",
            "            self.svals[slot] = value", "            self.scalar_ready.add(uid)", "            return value",
            "        finally:", "            self.active.remove(active_key)", "",
            "    def reduction(self, uid):",
            "        row = _PHYSICAL[uid]", "        carried = row['carried']", "        producer = row['producer_stage']",
            "        if carried is not None and self.current_stage is not None and producer is not None and self.current_stage > producer:",
            "            if not self.carried_valid[carried]:",
            "                raise RuntimeError(f'stage-carried reduction {uid} unavailable in stage {self.current_stage}')",
            "            return self.carried[carried]",
            "        slot = row['scalar']", "        if uid not in self.scalar_ready:",
            "            raise RuntimeError(f'reduction {uid} requested before direct stage scan completion')",
            "        return self.svals[slot]", "",
            "    def _geometry(self, stage=None):",
            "        stage = self.current_stage if stage is None else int(stage)",
            "        if stage not in self.stage_geometry:",
            "            raise RuntimeError(f'Stage geometry {stage} is not initialized')",
            "        return self.stage_geometry[stage]", "",
            "    def _position(self, q, stage=None):",
            "        start, _stop, step, T, _base_start, _base_stop, _base_T = self._geometry(stage)",
            "        delta = int(q) - start", "        if step == 0 or delta % step != 0:", "            return None",
            "        p = delta // step", "        return p if 0 <= p < T else None", "",
            "    def _read_ring(self, row, q, main_axis=True):",
            "        if main_axis:", "            p = self._position(q)", "            if p is None:", "                return False, None",
            "            offset = p", "        else:", "            offset = int(q)",
            "        slot = row['ring_base'] + (offset % row['ring_depth'])",
            "        if self.ring_coord[slot] == q:", "            return True, self.ring[slot]",
            "        return False, None", "",
            "    def _store_ring(self, row, q, value, main_axis=True):",
            "        if main_axis:", "            p = self._position(q)", "            if p is None:", "                raise RuntimeError(f'ring coordinate {q} outside direct Stage domain')",
            "            offset = p", "        else:", "            offset = int(q)",
            "        slot = row['ring_base'] + (offset % row['ring_depth'])",
            "        self.ring[slot] = value", "        self.ring_coord[slot] = q", "",
            "    def auxiliary(self, uid, q):",
            "        row = _PHYSICAL[uid]", "        if row['kind'] != 'ring':", "            raise RuntimeError(f'auxiliary recurrence {uid} lacks ring storage')",
            "        ok, value = self._read_ring(row, q, main_axis=False)", "        if ok:", "            return value",
            "        active_key = ('aux', uid, q)", "        if active_key in self.active:",
            "            raise RuntimeError(f'auxiliary recurrence same-coordinate cycle: {uid}({q})')",
            "        self.active.add(active_key)", "        try:", "            value = self._invoke(uid, (q,))",
            "            self._store_ring(row, q, value, main_axis=False)", "            return value",
            "        finally:", "            self.active.remove(active_key)", "",
            "    def _read_materialized(self, uid, q):",
            "        row = _PHYSICAL[uid]", "        slot = row['materialized']", "        producer = row['producer_stage']",
            "        if slot is None or producer is None or self.materialized[slot] is None:", "            return False, None",
            "        p = self._position(q, producer)", "        if p is None or not self.materialized_valid[slot][p]:", "            return False, None",
            "        return True, self.materialized[slot][p]", "",
            "    def _store_materialized(self, uid, q, value):",
            "        row = _PHYSICAL[uid]", "        slot = row['materialized']", "        producer = row['producer_stage']",
            "        if slot is None or producer is None or self.current_stage != producer:", "            return",
            "        p = self._position(q, producer)",
            "        if p is None:", "            raise RuntimeError(f'materialized value {uid}({q}) outside producer Stage {producer} domain')",
            "        self.materialized[slot][p] = value", "        self.materialized_valid[slot][p] = True", "",
            "    def _read_persistent(self, uid, q):",
            "        row = _PHYSICAL[uid]", "        producer = row['producer_stage']",
            "        if row['materialized'] is not None and self.current_stage is not None and producer is not None and self.current_stage > producer:",
            "            ok, value = self._read_materialized(uid, q)", "            if ok:", "                return True, value",
            "            return False, None",
            "        seed_key = (uid, q)",
            "        if seed_key in self.replay_values:", "            return True, self.replay_values[seed_key]",
            "        seed_spec = self.seed_specs.get((self.current_stage, uid, q))",
            "        if seed_spec is not None:",
            "            active_boundaries, replay_start = seed_spec", "            start, _stop, _step, _T, _base_start, _base_stop, _base_T = self._geometry()",
            "            if not active_boundaries or start in active_boundaries:",
            "                if seed_key not in self.seeds:", "                    self._realize_boundary_seed(uid, q, replay_start)",
            "                return True, self.seeds[seed_key]",
            "        kind = row['kind']",
            "        if kind == 'current':", "            slot = row['current']", "            if self.current_coord[slot] == q:", "                return True, self.current[slot]",
            "        elif kind == 'ring':", "            return self._read_ring(row, q, main_axis=True)",
            "        elif kind == 'history':", "            p = self._position(q)",
            "            if p is not None and self.history is not None and self.history_valid is not None:",
            "                slot = row['history']", "                if self.history_valid[slot, p]:", "                    return True, self.history[slot, p]",
            "        return False, None", "",
            "    def _realize_boundary_seed(self, uid, q, replay_start):",
            "        saved_q = self.current_q", "        try:",
            "            if replay_start is None:",
            "                self.seeds[(uid, q)] = self._invoke(uid, (q,))", "                return",
            "            step = 1 if replay_start <= q else -1", "            value = None",
            "            for k in range(int(replay_start), int(q) + step, step):",
            "                self.current_q = k", "                self.coordinate_cache.clear()",
            "                value = self._invoke(uid, (k,))", "                self.replay_values[(uid, k)] = value",
            "            self.seeds[(uid, q)] = value",
            "        finally:",
            "            self.current_q = saved_q",
            "            for key in [x for x in self.replay_values if x[0] == uid]:", "                del self.replay_values[key]", "",
            "    def _store_persistent(self, uid, q, value):",
            "        row = _PHYSICAL[uid]", "        kind = row['kind']",
            "        if kind == 'current':", "            slot = row['current']", "            self.current[slot] = value", "            self.current_coord[slot] = q",
            "        elif kind == 'ring':", "            self._store_ring(row, q, value, main_axis=True)",
            "        elif kind == 'history':", "            p = self._position(q)",
            "            if p is None:", "                raise RuntimeError(f'history value {uid} coordinate {q} outside direct Stage domain')",
            "            self.history[row['history'], p] = value", "            self.history_valid[row['history'], p] = True",
            "        else:", "            raise RuntimeError(f'persistent value {uid} has unsupported physical storage {kind}')",
            "        self._store_materialized(uid, q, value)", "",
            "    def persistent(self, uid, q):",
            "        ok, value = self._read_persistent(uid, q)", "        if ok:", "            return value",
            "        producer = _PHYSICAL[uid]['producer_stage']",
            "        if producer is not None and self.current_stage is not None and self.current_stage > producer:",
            "            raise RuntimeError(f'materialized persistent value {uid}({q}) unavailable in consumer Stage {self.current_stage}')",
            "        if self.current_q is None or q != self.current_q:",
            "            raise RuntimeError(f'persistent value {uid}({q}) unavailable at current coordinate {self.current_q}')",
            "        row = _PHYSICAL[uid]", "        domain = row['domain']",
            "        if domain is not None and not (domain[0] <= q <= domain[1]):",
            "            raise RuntimeError(f'persistent value {uid}({q}) outside proven canonical domain {domain}')",
            "        active_key = ('persistent', uid, q)", "        if active_key in self.active:",
            "            raise RuntimeError(f'same-coordinate persistent recursion cycle: {uid}({q})')",
            "        self.active.add(active_key)", "        try:", "            value = self._invoke(uid, (q,))",
            "            self._store_persistent(uid, q, value)", "            return value",
            "        finally:", "            self.active.remove(active_key)", "",
            "    def _barrier_row(self, uid):",
            "        for row in _BARRIERS:",
            "            if row[0] == uid:", "                return row",
            "        raise RuntimeError(f'unknown physical completion barrier {uid}')", "",
            "    def _verify_materialized_history_complete(self, barrier_uid, uid):",
            "        row = _PHYSICAL[uid]", "        slot = row['materialized']", "        producer = row['producer_stage']",
            "        if slot is None or producer is None:", "            raise RuntimeError(f'barrier {barrier_uid} requires non-materialized value {uid}')",
            "        values = self.materialized[slot]", "        valid = self.materialized_valid[slot]",
            "        if values is None or valid is None:", "            raise RuntimeError(f'barrier {barrier_uid} materialized history {uid} was not allocated')",
            "        bounds = row['retained_bounds']",
            "        if bounds is None:", "            raise RuntimeError(f'barrier {barrier_uid} lacks retained bounds for {uid}')",
            "        lo, hi = bounds", "        start, stop, step, T, _bs, _be, _bt = self._geometry(producer)",
            "        if step <= 0:", "            raise RuntimeError(f'barrier {barrier_uid} retained-domain verification requires positive Stage step')",
            "        lo = start if lo is None else int(lo)", "        hi = (stop - step) if hi is None else int(hi)",
            "        domain = row['domain']",
            "        if domain is not None:", "            lo = max(lo, int(domain[0]))", "            hi = min(hi, int(domain[1]))",
            "        first = start if start >= lo else start + ((lo - start + step - 1) // step) * step",
            "        last = (stop - step) if (stop - step) <= hi else hi - ((hi - start) % step)",
            "        if first <= last:",
            "            if first < start or last >= stop:", "                raise RuntimeError(f'barrier {barrier_uid} retained domain for {uid} exceeds producer Stage geometry')",
            "            p0 = (first - start) // step", "            p1 = (last - start) // step",
            "            if p0 < 0 or p1 >= T:", "                raise RuntimeError(f'barrier {barrier_uid} retained positions for {uid} exceed producer Stage storage')",
            "            if not bool(np.all(valid[p0:p1 + 1])):", "                missing = [start + p * step for p in range(p0, p1 + 1) if not valid[p]]",
            "                raise RuntimeError(f'barrier {barrier_uid} materialized history {uid} incomplete at coordinates {missing[:16]}')", "",
            "    def _satisfy_outgoing_barriers(self, stage):",
            "        for uid in _OUTGOING_BARRIERS.get(stage, ()):",
            "            barrier_uid, source, target, availability, execution_mode, required_uids, required_slots, _proof, _evidence, _mode_proof = self._barrier_row(uid)",
            "            if source != stage:", "                raise RuntimeError(f'barrier {uid} source Stage mismatch: {source} != {stage}')",
            "            if execution_mode != 'full_stage_materialization':", "                raise RuntimeError(f'unsupported full-stage fence execution mode {execution_mode}')",
            "            if tuple(_PHYSICAL[x]['materialized'] for x in required_uids) != tuple(required_slots):",
            "                raise RuntimeError(f'barrier {uid} materialized slot identity changed')",
            "            for value_uid in required_uids:", "                self._verify_materialized_history_complete(uid, value_uid)",
            "            self.satisfied_barriers.add(uid)", "",
            "    def _require_incoming_barriers(self, stage):",
            "        for uid in _INCOMING_BARRIERS.get(stage, ()):",
            "            if uid not in self.satisfied_barriers:", "                raise RuntimeError(f'Stage {stage} entered before completion barrier {uid} was satisfied')", "",
            "    def _prepare_stage(self, stage):",
            "        self._require_incoming_barriers(int(stage))",
            "        self.current_stage = int(stage)", "        self.current_q = None", "        self.coordinate_cache.clear()",
            "        if _POST_STAGE_SCALARS.get(stage):", "            return",
            "        args = tuple(int(x) for x in globals()[f'_direct_range_args_stage_{stage}']())",
            "        base = range(*args)",
            "        frozen = next(x for x in _STAGE_DOMAINS if x[0] == stage)", "        expected_step = int(frozen[3])",
            "        ext_min, ext_max = int(frozen[5]), int(frozen[6])",
            "        if int(base.step) != expected_step:",
            "            raise RuntimeError(f'runtime coordinate step {base.step} contradicts frozen Stage {stage} contract {expected_step}')",
            "        if base.step <= 0 and (ext_min or ext_max):",
            "            raise RuntimeError(f'materialization extension for nonascending range is outside direct subset: Stage {stage}')",
            "        start = int(base.start) + ext_min", "        stop = int(base.stop) + ext_max",
            "        r = range(start, stop, int(base.step))",
            "        self.stage_geometry[stage] = (int(r.start), int(r.stop), int(r.step), len(r), int(base.start), int(base.stop), len(base))",
            "        T = len(r)",
            "        self.history = np.empty((_HISTORY_SLOT_COUNT, max(1, T)), dtype=np.float64) if _HISTORY_SLOT_COUNT else None",
            "        self.history_valid = np.zeros((_HISTORY_SLOT_COUNT, max(1, T)), dtype=np.bool_) if _HISTORY_SLOT_COUNT else None",
            "        for uid in _MATERIALIZED_BY_STAGE.get(stage, ()):",
            "            slot = _PHYSICAL[uid]['materialized']",
            "            self.materialized[slot] = np.empty(max(1, T), dtype=np.float64)",
            "            self.materialized_valid[slot] = np.zeros(max(1, T), dtype=np.bool_)", "",
            "    def _reset_main_coordinate_storage(self):",
            "        for uid, row in _PHYSICAL.items():",
            "            if uid in _AUXILIARY:", "                continue",
            "            if row['producer_stage'] is not None and row['producer_stage'] != self.current_stage:", "                continue",
            "            if row['current'] is not None:", "                self.current_coord[row['current']] = None",
            "            if row['ring_base'] is not None:",
            "                for j in range(row['ring_base'], row['ring_base'] + row['ring_depth']):", "                    self.ring_coord[j] = None",
            "        if self.history_valid is not None:", "            self.history_valid.fill(False)",
            "        self.coordinate_cache.clear()", "        self.current_q = None", "",
            "    def _finalize_stage(self, stage):",
            "        for uid in _CARRIED_BY_STAGE.get(stage, ()):",
            "            row = _PHYSICAL[uid]", "            slot = row['scalar']", "            carried = row['carried']",
            "            if uid not in self.scalar_ready:", "                self.scalar(uid)",
            "            self.carried[carried] = self.svals[slot]", "            self.carried_valid[carried] = True", "",
            "    def scan(self):",
            "        for stage in _STAGE_ORDER:",
            "            self._prepare_stage(stage)",
            "            if _POST_STAGE_SCALARS.get(stage):",
            "                for uid in _POST_STAGE_SCALARS[stage]:", "                    self.scalar(uid)",
            "                self._finalize_stage(stage)",
            "                self._satisfy_outgoing_barriers(stage)",
            "                continue",
            "            for block_stage, _domain_uid, drivers, reductions, scan_direction in _BLOCKS:",
            "                if block_stage != stage:", "                    continue",
            "                self._reset_main_coordinate_storage()",
            "                for uid in reductions:",
            "                    slot = _PHYSICAL[uid]['scalar']", "                    self.svals[slot] = globals()[f'_red_init_{uid}']()", "                    self.scalar_ready.discard(uid)",
            "                _start, _stop, _step, T, base_start, base_stop, base_T = self._geometry(stage)",
            "                if scan_direction == 'ascending':", "                    positions = range(T)",
            "                elif scan_direction == 'descending':", "                    positions = range(T - 1, -1, -1)",
            "                else:", "                    raise RuntimeError(f'unsupported direct scan direction {scan_direction}')",
            "                for pos in positions:",
            "                    start, _stop, step, _T, _base_start, _base_stop, _base_T = self._geometry(stage)", "                    q = start + pos * step",
            "                    self.current_q = q", "                    self.coordinate_cache.clear()",
            "                    for uid in drivers:",
            "                        domain = _PHYSICAL[uid]['domain']",
            "                        if domain is None or domain[0] <= q <= domain[1]:", "                            self.persistent(uid, q)",
            "                    delta_base = q - base_start",
            "                    in_base = (_step != 0 and delta_base % _step == 0 and 0 <= delta_base // _step < base_T)",
            "                    if in_base:",
            "                        for uid in reductions:",
            "                            slot = _PHYSICAL[uid]['scalar']", "                            acc = self.svals[slot]", "                            filt = globals()[f'_red_filter_{uid}']",
            "                            if filt(q, acc):", "                                self.svals[slot] = acc + globals()[f'_red_body_{uid}'](q, acc)",
            "                self.current_q = None",
            "                for uid in reductions:", "                    self.scalar_ready.add(uid)",
            "            self._finalize_stage(stage)",
            "            self._satisfy_outgoing_barriers(stage)",
            "        self.current_stage = None", "        self.current_q = None", "",
            "    def result(self):", f"        return float(self.call({p.output_uid!r}, ()))", "",
            "def _run_one(point_index, inputs):",
            "    global _ACTIVE", "    if _ACTIVE is not None:", "        raise RuntimeError('direct Stage kernel is not reentrant')",
            "    kernel = _StageKernel(point_index, inputs)", "    _ACTIVE = kernel", "    try:",
            "        kernel.scan()", "        return kernel.result()",
            "    finally:", "        _ACTIVE = None", "",
        ]
        return lines

    def _validation_lines(self, indent: str = "    ") -> list[str]:
        lines: list[str] = []
        for key, spec in self.inputs.items():
            if spec.enum_labels is not None:
                lines += [
                    f"{indent}_enum_{key} = np.asarray(inputs[{key!r}])",
                    f"{indent}if np.any(_enum_{key} < -1) or np.any(_enum_{key} >= {len(spec.enum_labels)}):",
                    f"{indent}    raise ValueError({('enum input ' + key + ' contains an invalid code')!r})",
                ]
        for fact in self.program.validated_static_facts:
            spec = self.inputs[fact.validation_input_key]
            if spec.enum_labels is not None:
                expected = spec.enum_labels.index(fact.value); expr = f"_enum_{fact.validation_input_key}"
            else:
                expected = int(fact.value) if spec.dtype == "int64" else bool(fact.value) if spec.dtype == "bool" else float(fact.value)
                expr = f"np.asarray(inputs[{fact.validation_input_key!r}])"
            lines += [
                f"{indent}if np.any({expr} != {expected!r}):",
                f"{indent}    raise ValueError({('validated static ' + fact.source_name + ' changed from proven value ' + repr(fact.value))!r})",
            ]
        return lines

    def python_source(self, module_name: str = "stage_graph_python") -> str:
        point_keys = [k for k, s in self.inputs.items() if s.scope == "point" and s.ndim == 1]
        n_expr = (
            f"len(inputs[{point_keys[0]!r}])"
            if point_keys else str(int(self.program.run_domain.point_count))
        )
        lines = self._core_lines() + ["def run(inputs):", f"    n = {n_expr}"]
        lines += self._validation_lines("    ")
        if point_keys:
            lines += [
                f"    if n != {int(self.program.run_domain.point_count)}:",
                f"        raise ValueError('point-scoped input length does not match frozen Stage run domain {int(self.program.run_domain.point_count)}')",
            ]
        lines += ["    return np.fromiter((_run_one(i, inputs) for i in range(n)), dtype=np.float64, count=n)", ""]
        return "\n".join(lines)

    @staticmethod
    def _ctype(dtype: str) -> str:
        if dtype == "float64": return "double"
        if dtype == "int64": return "long"
        if dtype == "bool": return "bint"
        raise CodegenError(f"unsupported direct Cython scalar dtype {dtype!r}")

    def _input_decl(self, key: str) -> str:
        s = self.inputs[key]
        if s.ndim == 0: return f"{self._ctype(s.dtype)} {key}"
        if s.dtype == "int64" and s.ndim == 1: return f"long[::1] {key}"
        if s.dtype == "bool" and s.ndim == 1: return f"cnp.npy_bool[::1] {key}"
        if s.dtype == "float64" and s.ndim == 1: return f"double[::1] {key}"
        if s.dtype == "float64" and s.ndim == 2: return f"double[:, ::1] {key}"
        if s.dtype == "int64" and s.ndim == 2: return f"long[:, ::1] {key}"
        if s.scope == "global" and s.ndim >= 2 and s.dtype in {"float64", "int64", "bool"}:
            return f"cnp.ndarray {key}"
        raise CodegenError(f"unsupported direct Cython input layout {key}: {s}")

    def cython_source(self, module_name: str = "stage_graph_template") -> str:
        if not self.program.direct_cython_supported:
            raise CodegenError("StageOptimizedProgram lacks direct Cython permission: " + "; ".join(self.program.direct_cython_blockers))
        point_keys = [k for k, s in self.inputs.items() if s.scope == "point" and s.ndim == 1]
        core = self._core_lines()
        core.insert(0, "# cython: language_level=3")
        core.insert(1, "cimport numpy as cnp")
        decls = [self._input_decl(k) for k in self.program.input_order]
        args = ",\n    ".join([*decls, "int threads=1"])
        input_dict = "{" + ", ".join(f"{k!r}: {k}" for k in self.program.input_order) + "}"
        n_expr = (
            f"{point_keys[0]}.shape[0]"
            if point_keys else str(int(self.program.run_domain.point_count))
        )
        lines = core + [
            "cpdef cnp.ndarray run(", f"    {args}", "):",
            f"    cdef Py_ssize_t n = {n_expr}", "    cdef Py_ssize_t i",
            "    cdef cnp.ndarray out = np.empty(n, dtype=np.float64)", f"    inputs = {input_dict}",
        ]
        lines += self._validation_lines("    ")
        if point_keys:
            lines += [
                f"    if n != {int(self.program.run_domain.point_count)}:",
                f"        raise ValueError('point-scoped input length does not match frozen Stage run domain {int(self.program.run_domain.point_count)}')",
            ]
        lines += ["    for i in range(n):", "        out[i] = _run_one(i, inputs)", "    return out", ""]
        return "\n".join(lines)
