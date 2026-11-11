from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping, Iterator
from dataclasses import dataclass, field
from typing import Any

from .loop_recovery import StructuredSequentialProgram
from .sequential_ir import SequentialError
from .storage import StoragePlan, StoragePlanError


class SlotStorageError(SequentialError):
    """Raised when physical slot lowering diverges from the realized schedule."""


@dataclass(frozen=True)
class NodeSlot:
    node_id: int
    formula_op_id: int
    slot_index: int


@dataclass(frozen=True)
class FormulaSlotLayout:
    formula_op_id: int
    name: str
    fullname: str
    storage_class: str
    produced_count: int
    slot_count: int
    ring_compatible: bool

    @property
    def slot_reduction(self) -> float:
        return self.produced_count / self.slot_count if self.slot_count else 1.0


@dataclass
class PhysicalSlotLayout:
    """Exact physical slot assignment for one realized StoragePlan.

    StoragePlan proves value lifetimes.  This pass turns that proof into concrete
    addresses.  No runtime release operation is required: when a slot becomes
    reusable, the next producer simply overwrites it after its formula body has
    consumed the previous owner.
    """

    plan: StoragePlan = field(repr=False)
    node_slots: tuple[NodeSlot, ...]
    formula_layouts: tuple[FormulaSlotLayout, ...]

    def __post_init__(self) -> None:
        self._node_slot_by_id = {x.node_id: x for x in self.node_slots}
        self._formula_layout_by_id = {x.formula_op_id: x for x in self.formula_layouts}

    @property
    def node_slot_by_id(self) -> dict[int, NodeSlot]:
        return self._node_slot_by_id

    @property
    def formula_layout_by_id(self) -> dict[int, FormulaSlotLayout]:
        return self._formula_layout_by_id

    @property
    def total_slots(self) -> int:
        return sum(x.slot_count for x in self.formula_layouts)

    @property
    def baseline_values(self) -> int:
        return self.plan.baseline_scheduled_values

    @property
    def physical_slot_reduction(self) -> float:
        return self.baseline_values / self.total_slots if self.total_slots else 1.0

    def validate(self) -> None:
        lifetimes = self.plan.lifetimes
        if tuple(x.node_id for x in self.node_slots) != tuple(x.node_id for x in lifetimes):
            raise SlotStorageError("physical slot order differs from storage lifetime order")
        by_formula: dict[int, list[NodeSlot]] = defaultdict(list)
        for ns in self.node_slots:
            by_formula[ns.formula_op_id].append(ns)
        for fl in self.formula_layouts:
            rows = by_formula.get(fl.formula_op_id, [])
            if len(rows) != fl.produced_count:
                raise SlotStorageError(
                    f"formula {fl.fullname} has {len(rows)} slot rows, expected {fl.produced_count}"
                )
            if rows and max(x.slot_index for x in rows) >= fl.slot_count:
                raise SlotStorageError(f"formula {fl.fullname} uses slot beyond declared capacity")


@dataclass(frozen=True)
class SlotExecutionStats:
    operation_count: int
    physical_slots: int
    baseline_scheduled_values: int
    slot_reads: int
    slot_writes: int
    hot_loop_deletes: int
    fallback_mapping_reads: int
    fallback_mapping_writes: int

    @property
    def physical_slot_reduction(self) -> float:
        return self.baseline_scheduled_values / self.physical_slots if self.physical_slots else 1.0


class _SlotRuntime:
    def __init__(self, layout: PhysicalSlotLayout, structured: StructuredSequentialProgram):
        self.layout = layout
        self.structured = structured
        max_fid = max((fl.formula_op_id for fl in layout.formula_layouts), default=-1)
        self.values: list[list[Any]] = [[] for _ in range(max_fid + 1)]
        self.owners: list[list[int | None]] = [[] for _ in range(max_fid + 1)]
        for fl in layout.formula_layouts:
            self.values[fl.formula_op_id] = [None] * fl.slot_count
            self.owners[fl.formula_op_id] = [None] * fl.slot_count
        self.slot_reads = 0
        self.slot_writes = 0
        self.hot_loop_deletes = 0
        self.fallback_mapping_reads = 0
        self.fallback_mapping_writes = 0

    def reset(self) -> None:
        # Do not clear value arrays.  Owners are the validity bits and every slot
        # required by this exact artifact is overwritten on the next run.  This
        # deliberately avoids an O(trace-size) cleanup pass between executions.
        for owner in self.owners:
            owner[:] = [None] * len(owner)
        self.slot_reads = 0
        self.slot_writes = 0
        self.hot_loop_deletes = 0
        self.fallback_mapping_reads = 0
        self.fallback_mapping_writes = 0

    def read_node(self, node_id: int) -> Any:
        addr = self.layout.node_slot_by_id[node_id]
        owners = self.owners[addr.formula_op_id]
        if owners[addr.slot_index] != node_id:
            raise KeyError(node_id)
        self.slot_reads += 1
        return self.values[addr.formula_op_id][addr.slot_index]

    def write_node(self, node_id: int, value: Any) -> None:
        addr = self.layout.node_slot_by_id[node_id]
        self.write_address(addr, value)

    def write_address(self, addr: NodeSlot, value: Any) -> None:
        self.values[addr.formula_op_id][addr.slot_index] = value
        self.owners[addr.formula_op_id][addr.slot_index] = addr.node_id
        self.slot_writes += 1

    def has_node(self, node_id: int) -> bool:
        addr = self.layout.node_slot_by_id[node_id]
        return self.owners[addr.formula_op_id][addr.slot_index] == node_id


class _SlotData(MutableMapping):
    """Dict-compatible view used by modelx CellsImpl during slot execution.

    Scheduled realized keys are backed by compiler-owned physical slots. Explicit
    inputs and any unscheduled/cache-bound values delegate to the original modelx
    dictionary.  The hot path never deletes scheduled keys; slot ownership changes
    when a later producer stores into the same address.
    """

    def __init__(self, runtime: _SlotRuntime, base: dict, key_to_node: dict[tuple[Any, ...], int]):
        self.runtime = runtime
        self.base = base
        self.key_to_node = key_to_node

    def __getitem__(self, key):
        nid = self.key_to_node.get(key)
        if nid is not None:
            return self.runtime.read_node(nid)
        self.runtime.fallback_mapping_reads += 1
        return self.base[key]

    def __setitem__(self, key, value):
        nid = self.key_to_node.get(key)
        if nid is not None:
            self.runtime.write_node(nid, value)
            return
        self.runtime.fallback_mapping_writes += 1
        self.base[key] = value

    def __delitem__(self, key):
        nid = self.key_to_node.get(key)
        if nid is not None:
            self.runtime.hot_loop_deletes += 1
            addr = self.runtime.layout.node_slot_by_id[nid]
            owners = self.runtime.owners[addr.formula_op_id]
            if owners[addr.slot_index] != nid:
                raise KeyError(key)
            owners[addr.slot_index] = None
            return
        del self.base[key]

    def __contains__(self, key):
        nid = self.key_to_node.get(key)
        if nid is not None:
            return self.runtime.has_node(nid)
        return key in self.base

    def __iter__(self) -> Iterator:
        # Mapping iteration is not expected in formula hot paths, but preserve the
        # observable mapping contract for modelx interfaces and diagnostics.
        yielded = set()
        for key in self.base:
            if key not in self.key_to_node:
                yielded.add(key)
                yield key
        for key, nid in self.key_to_node.items():
            if key not in yielded and self.runtime.has_node(nid):
                yield key

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


@dataclass
class SlotStorageProgram:
    """Stage-4a physical-slot executor.

    Formula semantics stay in the original Python/modelx functions.  The storage
    implementation, however, is no longer a growing key dictionary with explicit
    last-use deletion.  Stage-3a lifetimes are lowered once into fixed addresses,
    then normal formula evaluation overwrites those addresses automatically.
    """

    structured: StructuredSequentialProgram = field(repr=False)
    storage_plan: StoragePlan
    layout: PhysicalSlotLayout
    prepared: bool = False

    def __post_init__(self) -> None:
        self.layout.validate()
        self._runtime = _SlotRuntime(self.layout, self.structured)
        self._base_data: dict[Any, dict] = {}
        self._key_to_node_by_obj: dict[Any, dict[tuple[Any, ...], int]] = defaultdict(dict)
        by_id = self.structured.sequential.node_by_id
        for lt in self.storage_plan.lifetimes:
            node = by_id[lt.node_id]
            self._key_to_node_by_obj[node.obj][node.args] = node.node_id

    @property
    def physical_slots(self) -> int:
        return self.layout.total_slots

    @property
    def physical_slot_reduction(self) -> float:
        return self.layout.physical_slot_reduction

    def prepare(self) -> None:
        """One-time transition from trace caches to reusable physical storage.

        This may clear the original realized cache once.  It is build/preparation
        work, not execution-loop work; subsequent runs issue zero scheduled-key
        deletes under the slot runtime.
        """
        if self.prepared:
            return
        seq = self.structured.sequential
        seq._clear_scheduled_values()
        for obj in self._key_to_node_by_obj:
            self._base_data[obj] = obj.data
        self.prepared = True

    def execute(self, *, validate: bool = False):
        if not self.prepared:
            self.prepare()
        self.storage_plan.validate()
        self.layout.validate()
        self._runtime.reset()

        seq = self.structured.sequential
        expected_ops = seq.ops
        by_id = seq.node_by_id
        installed: list[tuple[Any, dict]] = []
        raw_events = None
        result = None

        try:
            for obj, keymap in self._key_to_node_by_obj.items():
                base = self._base_data[obj]
                installed.append((obj, obj.data))
                obj.data = _SlotData(self._runtime, base, keymap)

            system = seq.trace.model._impl.system
            if validate:
                with system.trace_stack(maxlen=None):
                    self._execute_ops(expected_ops, by_id)
                    result = self._read_targets()
                    raw_events = list(system.callstack.tracestack)
            else:
                self._execute_ops(expected_ops, by_id)
                result = self._read_targets()

            if self._runtime.hot_loop_deletes:
                raise SlotStorageError(
                    f"slot execution issued {self._runtime.hot_loop_deletes} scheduled-key deletes"
                )

            if validate:
                replay = seq._validate_replay(raw_events or [], direct=True, result=result)
                if not replay.ok:
                    raise SlotStorageError(
                        "slot-backed replay diverged from realized schedule: " + "; ".join(replay.reasons)
                    )
            else:
                replay = None

            stats = SlotExecutionStats(
                operation_count=len(expected_ops),
                physical_slots=self.layout.total_slots,
                baseline_scheduled_values=self.storage_plan.baseline_scheduled_values,
                slot_reads=self._runtime.slot_reads,
                slot_writes=self._runtime.slot_writes,
                hot_loop_deletes=self._runtime.hot_loop_deletes,
                fallback_mapping_reads=self._runtime.fallback_mapping_reads,
                fallback_mapping_writes=self._runtime.fallback_mapping_writes,
            )
            if validate:
                return result, replay, stats
            return result, stats
        finally:
            # Preserve only end-pinned realized results in the ordinary modelx
            # dictionaries. Everything else remains compiler-owned/transient.
            retained_values: list[tuple[Any, tuple[Any, ...], Any]] = []
            if result is not None:
                for nid in self.storage_plan.retained_node_ids:
                    node = by_id[nid]
                    try:
                        value = self._runtime.read_node(nid)
                    except KeyError:
                        continue
                    retained_values.append((node.obj, node.args, value))
            for obj, original in reversed(installed):
                obj.data = original
            for obj, key, value in retained_values:
                obj._store_value(key, value)

    def _execute_ops(self, expected_ops, by_id) -> None:
        count = 0
        for pos, runtime_node in enumerate(self.structured.iter_canonical_runtime_nodes()):
            if pos >= len(expected_ops):
                raise SlotStorageError("canonical slot executor produced extra operation")
            nid = expected_ops[pos].node_id
            expected = by_id[nid]
            if runtime_node[0] is not expected.obj or runtime_node[1] != expected.args:
                raise SlotStorageError(f"canonical runtime node differs at operation {pos}")
            obj, key = runtime_node
            if key in obj.input_keys:
                raise SlotStorageError("scheduled formula changed Cell value into explicit input")
            # on_eval_formula stores through obj.data. While the slot mapping is
            # installed, _store_value therefore writes directly to the assigned
            # physical address and automatically overwrites a dead predecessor.
            obj.on_eval_formula(key)
            count += 1
        if count != len(expected_ops):
            raise SlotStorageError(
                f"canonical slot executor produced {count} operations, expected {len(expected_ops)}"
            )

    def _read_targets(self):
        vals = []
        runtime_to_id = self.structured.sequential.trace.runtime_to_id
        for obj, key in self.structured.sequential.trace.target_runtime_nodes:
            nid = runtime_to_id.get((obj, tuple(key)))
            if nid is not None and nid in self.layout.node_slot_by_id:
                try:
                    vals.append(self._runtime.read_node(nid))
                    continue
                except KeyError:
                    pass
            vals.append(obj.get_value_from_key(tuple(key)))
        return vals[0] if len(vals) == 1 else tuple(vals)


def _effective_last_use(last_use_position: int, nops: int) -> int:
    return nops + 1 if last_use_position < 0 else last_use_position


def build_physical_slot_layout(plan: StoragePlan) -> PhysicalSlotLayout:
    plan.validate()
    nops = len(plan.lifetimes)
    specs = plan.formula_by_id
    grouped: dict[int, list[Any]] = defaultdict(list)
    for lt in plan.lifetimes:
        grouped[lt.formula_op_id].append(lt)

    addresses: dict[int, int] = {}
    formula_layouts: list[FormulaSlotLayout] = []

    for fid in sorted(grouped):
        rows = grouped[fid]
        spec = specs[fid]
        capacity = max(1, spec.required_slots)

        if spec.storage_class == "bounded_ring":
            for occ, lt in enumerate(rows):
                slot = occ % capacity
                if occ >= capacity:
                    prior = rows[occ - capacity]
                    if _effective_last_use(prior.last_use_position, nops) > lt.position:
                        raise SlotStorageError(
                            f"ring proof failed during physical lowering for {spec.fullname}"
                        )
                addresses[lt.node_id] = slot

        elif spec.storage_class in ("scalar", "immediate"):
            for lt in rows:
                addresses[lt.node_id] = 0

        elif spec.storage_class == "full_history":
            # Full history is deliberately left alone at this stage. Each realized
            # occurrence receives a stable array position, ready for typed Cython
            # storage without any dictionary lifetime bookkeeping.
            for occ, lt in enumerate(rows):
                addresses[lt.node_id] = occ
            capacity = len(rows)

        elif spec.storage_class == "bounded_slots":
            # Exact interval coloring. A slot whose previous owner's final use is
            # the current operation is reusable because the formula consumes the
            # predecessor before its result is stored.
            slot_end: list[int] = []
            for lt in rows:
                reusable = None
                for slot, end in enumerate(slot_end):
                    if end <= lt.position:
                        reusable = slot
                        break
                if reusable is None:
                    reusable = len(slot_end)
                    slot_end.append(-1)
                addresses[lt.node_id] = reusable
                slot_end[reusable] = _effective_last_use(lt.last_use_position, nops)
            capacity = len(slot_end)
            if capacity != spec.required_slots:
                raise SlotStorageError(
                    f"bounded-slot coloring for {spec.fullname} needs {capacity}, "
                    f"storage proof declared {spec.required_slots}"
                )
        else:
            raise SlotStorageError(f"unknown storage class {spec.storage_class!r}")

        formula_layouts.append(
            FormulaSlotLayout(
                formula_op_id=fid,
                name=spec.name,
                fullname=spec.fullname,
                storage_class=spec.storage_class,
                produced_count=spec.produced_count,
                slot_count=capacity,
                ring_compatible=spec.ring_compatible,
            )
        )

    node_slots = tuple(
        NodeSlot(lt.node_id, lt.formula_op_id, addresses[lt.node_id]) for lt in plan.lifetimes
    )
    result = PhysicalSlotLayout(plan=plan, node_slots=node_slots, formula_layouts=tuple(formula_layouts))
    result.validate()
    return result


def lower_to_slots(structured: StructuredSequentialProgram, plan: StoragePlan) -> SlotStorageProgram:
    if plan.structured is not structured:
        raise StoragePlanError("storage plan belongs to a different structured program")
    return SlotStorageProgram(
        structured=structured,
        storage_plan=plan,
        layout=build_physical_slot_layout(plan),
    )

# ---------------------------------------------------------------------------
# Direct-slot Python lowering
# ---------------------------------------------------------------------------

import dis
import inspect
import types

try:
    from modelx.core.cells import Cells as _MxCells
    from modelx.core.space import BaseSpace as _MxBaseSpace
    from modelx.core.execution.trace import get_node as _mx_get_node, KEY as _MX_KEY, tuplize_key as _mx_tuplize_key
except Exception:  # pragma: no cover - modelx is a required runtime dependency in this package
    _MxCells = ()
    _MxBaseSpace = ()
    _mx_get_node = None
    _MX_KEY = 1
    _mx_tuplize_key = None


class _DirectCellCallable:
    __slots__ = ("runtime", "impl", "original")

    def __init__(self, runtime, impl, original):
        self.runtime = runtime
        self.impl = impl
        self.original = original

    def __call__(self, *args, **kwargs):
        return self.runtime.read_cell(self.impl, args, kwargs, self.original)


class _DirectCellsProxy:
    __slots__ = ("runtime", "original", "impl")

    def __init__(self, runtime, original):
        self.runtime = runtime
        self.original = original
        self.impl = original._impl

    def __call__(self, *args, **kwargs):
        return self.runtime.read_cell(self.impl, args, kwargs, self.original)

    def __getitem__(self, key):
        if _mx_tuplize_key is None:
            return self.original[key]
        args = _mx_tuplize_key(self.original, key)
        return self.runtime.read_cell(self.impl, args, {}, lambda *a, **k: self.original[key])

    def __getattr__(self, name):
        # Methods such as match()/is_input() retain original modelx semantics. If
        # they read a scheduled key indirectly, the installed slot mapping is the
        # correctness bridge and the benchmark records that fallback explicitly.
        return getattr(self.original, name)

    def __iter__(self):
        return iter(self.original)

    def __len__(self):
        return len(self.original)


class _DirectSpaceProxy:
    __slots__ = ("runtime", "original")

    def __init__(self, runtime, original):
        self.runtime = runtime
        self.original = original

    def __getattr__(self, name):
        return self.runtime.wrap_reference(getattr(self.original, name))

    def __getitem__(self, key):
        return self.runtime.wrap_reference(self.original[key])

    def __call__(self, *args, **kwargs):
        return self.runtime.wrap_reference(self.original(*args, **kwargs))

    def __iter__(self):
        return iter(self.original)

    def __len__(self):
        return len(self.original)


class _DirectSlotRuntime(_SlotRuntime):
    def __init__(self, layout: PhysicalSlotLayout, structured: StructuredSequentialProgram):
        super().__init__(layout, structured)
        self.runtime_to_id = structured.sequential.trace.runtime_to_id
        self.address_by_obj: dict[Any, dict[tuple[Any, ...], NodeSlot]] = defaultdict(dict)
        node_by_id = structured.sequential.node_by_id
        for addr in layout.node_slots:
            node = node_by_id[addr.node_id]
            self.address_by_obj[node.obj][node.args] = addr
        self.direct_slot_reads = 0
        self.delegated_cell_reads = 0
        self._function_cache: dict[Any, Any] = {}
        self._reference_cache: dict[int, Any] = {}

    def reset(self) -> None:
        super().reset()
        self.direct_slot_reads = 0
        self.delegated_cell_reads = 0

    def read_cell(self, impl, args, kwargs, original):
        # Fast path for the overwhelmingly common positional realized call. The
        # trace keys are already canonical tuples, so avoid modelx signature
        # binding unless the direct lookup misses or kwargs are present.
        key = tuple(args) if not kwargs else None
        addr = None
        if key is not None:
            try:
                addr = self.address_by_obj.get(impl, {}).get(key)
            except TypeError:
                # Uncached helpers may legitimately receive unhashable Python
                # objects such as pandas Series. They remain ordinary Python calls.
                key = None
        if addr is None and key is not None and _mx_get_node is not None:
            try:
                key = tuple(_mx_get_node(impl, args, kwargs)[_MX_KEY])
                addr = self.address_by_obj.get(impl, {}).get(key)
            except Exception:
                addr = None
        if addr is not None:
            owners = self.owners[addr.formula_op_id]
            if owners[addr.slot_index] == addr.node_id:
                self.direct_slot_reads += 1
                self.slot_reads += 1
                return self.values[addr.formula_op_id][addr.slot_index]
            # Known realized node requested before its slot is available. Do not
            # silently substitute a stale modelx value. Fall through so strict
            # trace validation exposes any unexpected recursive evaluation.
        self.delegated_cell_reads += 1
        return original(*args, **kwargs)

    def wrap_reference(self, value):
        # Bound CellsImpl.call methods are the common representation for names
        # rebound by modelx's altered-function machinery.
        if inspect.ismethod(value):
            owner = getattr(value, "__self__", None)
            func = getattr(value, "__func__", None)
            if owner is not None and getattr(func, "__name__", None) == "call" and hasattr(owner, "data"):
                cache_key = id(value)
                cached = self._reference_cache.get(cache_key)
                if cached is None:
                    cached = _DirectCellCallable(self, owner, value)
                    self._reference_cache[cache_key] = cached
                return cached
        if _MxCells and isinstance(value, _MxCells):
            cache_key = id(value)
            cached = self._reference_cache.get(cache_key)
            if cached is None:
                cached = _DirectCellsProxy(self, value)
                self._reference_cache[cache_key] = cached
            return cached
        if _MxBaseSpace and isinstance(value, _MxBaseSpace):
            cache_key = id(value)
            cached = self._reference_cache.get(cache_key)
            if cached is None:
                cached = _DirectSpaceProxy(self, value)
                self._reference_cache[cache_key] = cached
            return cached
        return value

    def function_for(self, obj):
        cached = self._function_cache.get(obj)
        if cached is not None:
            return cached
        fn = obj.altfunc
        # A copied globals dictionary would change the semantics of explicit
        # global writes/deletes. Keep such unusual formulas on the conservative
        # original-function boundary; the installed fixed-slot mapping still
        # provides exact bounded storage for any scheduled Cell reads they make.
        if any(ins.opname in {"STORE_GLOBAL", "DELETE_GLOBAL"} for ins in dis.get_instructions(fn)):
            self._function_cache[obj] = fn
            return fn
        glb = dict(fn.__globals__)
        # Only names the bytecode can actually load need wrapping. This keeps
        # cloning cheap even when a model formula module has a large namespace.
        for name in set(fn.__code__.co_names):
            if name in glb:
                glb[name] = self.wrap_reference(glb[name])
        cloned = types.FunctionType(fn.__code__, glb, fn.__name__, fn.__defaults__, fn.__closure__)
        cloned.__kwdefaults__ = getattr(fn, "__kwdefaults__", None)
        cloned.__annotations__ = getattr(fn, "__annotations__", {}).copy()
        self._function_cache[obj] = cloned
        return cloned


@dataclass(frozen=True)
class DirectSlotExecutionStats:
    operation_count: int
    physical_slots: int
    baseline_scheduled_values: int
    direct_slot_reads: int
    delegated_cell_reads: int
    fallback_mapping_reads: int
    fallback_mapping_writes: int
    slot_writes: int
    hot_loop_deletes: int

    @property
    def physical_slot_reduction(self) -> float:
        return self.baseline_scheduled_values / self.physical_slots if self.physical_slots else 1.0

    @property
    def direct_read_fraction(self) -> float:
        total = self.direct_slot_reads + self.fallback_mapping_reads
        return self.direct_slot_reads / total if total else 1.0


@dataclass
class DirectSlotProgram:
    """Direct Python formula execution over compiler-owned fixed slots.

    Formula bytecode/source is unchanged. Only modelx Cell/Space references in
    its globals are rebound to readers backed by the physical StoragePlan. This
    removes modelx executor/cache lookup from ordinary realized dependencies while
    preserving arbitrary Python inside formulas. A slot-backed data mapping stays
    installed solely as a conservative bridge for indirect references that cannot
    be rebound from function globals.
    """

    slot_program: SlotStorageProgram = field(repr=False)

    def __post_init__(self) -> None:
        self.slot_program.prepare()
        self.structured = self.slot_program.structured
        self.storage_plan = self.slot_program.storage_plan
        self.layout = self.slot_program.layout
        self._runtime = _DirectSlotRuntime(self.layout, self.structured)
        self._base_data = self.slot_program._base_data
        self._key_to_node_by_obj = self.slot_program._key_to_node_by_obj
        # Rebinding is compilation work. Do it once here so runtime benchmarks
        # and production execution contain only formula/slot work.
        for obj in self._key_to_node_by_obj:
            self._runtime.function_for(obj)
        self._position_addresses = tuple(
            self.layout.node_slot_by_id[op.node_id] for op in self.structured.sequential.ops
        )

    @property
    def physical_slots(self) -> int:
        return self.layout.total_slots

    def execute(self, *, validate: bool = False):
        self.storage_plan.validate()
        self.layout.validate()
        self._runtime.reset()
        seq = self.structured.sequential
        by_id = seq.node_by_id
        expected_ops = seq.ops
        installed: list[tuple[Any, Any]] = []
        raw_events = None
        result = None
        try:
            # Safety bridge only. Direct global Cell references bypass this mapping
            # entirely; indirect/dynamic modelx paths can still read produced slots.
            for obj, keymap in self._key_to_node_by_obj.items():
                installed.append((obj, obj.data))
                obj.data = _SlotData(self._runtime, self._base_data[obj], keymap)

            system = seq.trace.model._impl.system
            if validate:
                with system.trace_stack(maxlen=None):
                    self._execute_ops(expected_ops, by_id)
                    result = self._read_targets()
                    raw_events = list(system.callstack.tracestack)
            else:
                self._execute_ops(expected_ops, by_id)
                result = self._read_targets()

            if self._runtime.hot_loop_deletes:
                raise SlotStorageError(
                    f"direct slot execution issued {self._runtime.hot_loop_deletes} scheduled-key deletes"
                )
            if validate:
                replay = seq._validate_replay(raw_events or [], direct=True, result=result)
                if not replay.ok:
                    raise SlotStorageError(
                        "direct-slot replay diverged from realized schedule: " + "; ".join(replay.reasons)
                    )
            else:
                replay = None
            stats = DirectSlotExecutionStats(
                operation_count=len(expected_ops),
                physical_slots=self.layout.total_slots,
                baseline_scheduled_values=self.storage_plan.baseline_scheduled_values,
                direct_slot_reads=self._runtime.direct_slot_reads,
                delegated_cell_reads=self._runtime.delegated_cell_reads,
                fallback_mapping_reads=self._runtime.fallback_mapping_reads,
                fallback_mapping_writes=self._runtime.fallback_mapping_writes,
                slot_writes=self._runtime.slot_writes,
                hot_loop_deletes=self._runtime.hot_loop_deletes,
            )
            if validate:
                return result, replay, stats
            return result, stats
        finally:
            retained_values: list[tuple[Any, tuple[Any, ...], Any]] = []
            if result is not None:
                for nid in self.storage_plan.retained_node_ids:
                    node = by_id[nid]
                    try:
                        retained_values.append((node.obj, node.args, self._runtime.read_node(nid)))
                    except KeyError:
                        pass
            for obj, original in reversed(installed):
                obj.data = original
            for obj, key, value in retained_values:
                obj._store_value(key, value)

    def _execute_ops(self, expected_ops, by_id):
        count = 0
        for pos, runtime_node in enumerate(self.structured.iter_canonical_runtime_nodes()):
            if pos >= len(expected_ops):
                raise SlotStorageError("direct canonical executor produced extra operation")
            nid = expected_ops[pos].node_id
            expected = by_id[nid]
            obj, key = runtime_node
            if obj is not expected.obj or key != expected.args:
                raise SlotStorageError(f"direct canonical runtime node differs at operation {pos}")
            if key in obj.input_keys:
                raise SlotStorageError("scheduled formula changed Cell value into explicit input")
            fn = self._runtime.function_for(obj)
            value = fn(*key)
            # Match CellsImpl._store_value's allow_none contract without writing
            # through the modelx dictionary/mapping hot path.
            if value is None and not obj.get_property("allow_none"):
                obj._store_value(key, value)  # raises modelx's canonical error
                raise AssertionError("unreachable")
            self._runtime.write_address(self._position_addresses[pos], value)
            count += 1
        if count != len(expected_ops):
            raise SlotStorageError(
                f"direct canonical executor produced {count} operations, expected {len(expected_ops)}"
            )

    def _read_targets(self):
        vals = []
        runtime_to_id = self.structured.sequential.trace.runtime_to_id
        for obj, key in self.structured.sequential.trace.target_runtime_nodes:
            nid = runtime_to_id.get((obj, tuple(key)))
            if nid is not None and nid in self.layout.node_slot_by_id:
                try:
                    vals.append(self._runtime.read_node(nid))
                    continue
                except KeyError:
                    pass
            vals.append(obj.get_value_from_key(tuple(key)))
        return vals[0] if len(vals) == 1 else tuple(vals)


def lower_to_direct_slots(slot_program: SlotStorageProgram) -> DirectSlotProgram:
    return DirectSlotProgram(slot_program=slot_program)
