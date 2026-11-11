from __future__ import annotations

import ast
import dis
import inspect
import re
from itertools import combinations
from collections import namedtuple
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .build import build_extension
from .loop_recovery import LiteralBlock
from .native_bindings import build_runtime_arg_sites, encode_runtime_bindings
from .native_family import (
    NativeFamilyError,
    _emit_output_index_expr,
    _emit_runtime_arg_expr,
    _emit_runtime_helpers,
    _emit_runtime_reference_helper,
    _exact_native_formula_ok,
    _RuntimeFormulaEmitter,
    ModelxCythonRuntimeFormulaEmitter,
    LiteralModelxCythonFormulaEmitter,
    modelx_cython_formula_native_ok,
    _family_output_structural_kinds,
    _hybrid_numeric_error,
    _hybrid_value_equal,
    _impl_from_global,
    _instance_output_descriptors,
    _typed_slot_plan,
)
from .native_plan import build_native_family_backend_plan, expand_native_family_instance
from .native_references import (
    _cell_call_name,
    _impl_from_formula_global as _reference_impl_from_formula_global,
    build_reference_binding_plan,
    encode_reference_instance,
    reference_structural_kinds,
)
from .realized_compiler import RealizedTraceCompiler
from .modelx_cython_formula_engine import (
    ModelxCythonFormulaCatalog,
    source_uses_modelx_caches,
)

try:
    from modelx.core.cells import Cells as _MxCells
    from modelx.core.space import BaseSpace as _MxBaseSpace, Namespace as _MxNamespace
    from modelx.core.execution.trace import get_node as _mx_get_node, KEY as _MX_KEY, tuplize_key as _mx_tuplize_key
except Exception:  # pragma: no cover
    _MxCells = ()
    _MxBaseSpace = ()
    _MxNamespace = ()
    _mx_get_node = None
    _MX_KEY = 1
    _mx_tuplize_key = None


class WholeArtifactError(RuntimeError):
    """Fail-closed error for non-recursive whole-artifact execution."""


@dataclass(frozen=True)
class WholeArtifactExecutionStats:
    operation_count: int
    family_operations: int
    literal_operations: int
    python_callback_operations: int
    native_operations: int
    modelx_trace_events: int | None
    exact: bool | None
    max_abs_error: float | None


@dataclass(frozen=True)
class WholeArtifactBuildReport:
    operation_count: int
    block_count: int
    family_count: int
    family_instance_count: int
    literal_block_count: int
    source_lines: int
    source_bytes: int
    native_formula_ops: int
    python_formula_ops: int
    native_operations: int
    python_operations: int
    pyx_path: str
    so_path: str
    integrated_register_region_count: int = 0
    integrated_register_operations: int = 0
    integrated_register_python_boundaries: int = 0
    integrated_register_fallback_count: int = 0
    frozen_reference_site_count: int = 0
    frozen_formula_op_count: int = 0
    frozen_boundary_occurrences: int = 0
    frozen_data_bytes: int = 0
    frozen_prepare_seconds: float = 0.0
    frozen_rejected_site_count: int = 0
    execution_profile: str = "manual"
    target_value_kinds: tuple[str, ...] = ()
    storage_baseline_shallow_bytes: int = 0
    storage_peak_shallow_bytes: int = 0
    storage_shallow_value_reduction: float = 1.0
    modelx_cython_formula_catalog_size: int = 0
    modelx_cython_formula_matched: int = 0
    modelx_cython_formula_native: int = 0
    modelx_cython_native_operations: int = 0
    modelx_cython_formula_rejections: tuple[tuple[int, str], ...] = ()
    frozen_static_object_formula_ops: int = 0
    frozen_static_object_operations: int = 0
    direct_role_dispatch_family_count: int = 0
    direct_role_dispatch_role_count: int = 0
    reference_structural_specialized_site_count: int = 0
    reference_structural_dynamic_site_count: int = 0
    formula_catalog_source: str = "none"


class _StrictCellResolver:
    """Read a Cell only from compiler-owned slots or explicit input state.

    There is deliberately no call to ``original`` on a miss. A scheduled value
    requested before its exact sequential predecessor has produced it is a
    compiler/schedule error, not an invitation for modelx to recurse.
    """

    __slots__ = (
        "mapping", "dslots", "islots", "bslots", "oslots",
        "downers", "iowners", "bowners", "oowners", "owner_checks",
        "input_values", "label", "impl",
    )

    def __init__(
        self,
        mapping: dict[tuple[Any, ...], Any],
        dslots: np.ndarray,
        islots: np.ndarray,
        bslots: np.ndarray,
        oslots: list[Any],
        downers: np.ndarray,
        iowners: np.ndarray,
        bowners: np.ndarray,
        oowners: np.ndarray,
        owner_checks: list[bool],
        input_values: dict[tuple[Any, ...], Any],
        label: str,
        impl: Any,
    ):
        self.mapping = mapping
        self.dslots = dslots
        self.islots = islots
        self.bslots = bslots
        self.oslots = oslots
        self.downers = downers
        self.iowners = iowners
        self.bowners = bowners
        self.oowners = oowners
        self.owner_checks = owner_checks
        self.input_values = input_values
        self.label = label
        self.impl = impl

    def _normalize(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
        if _mx_get_node is None:
            if kwargs:
                raise WholeArtifactError(
                    f"keyword Cell access is unsupported in strict artifact: {self.label}"
                )
            return tuple(args)
        try:
            return tuple(_mx_get_node(self.impl, args, kwargs)[_MX_KEY])
        except Exception as exc:
            raise WholeArtifactError(
                f"Cell key cannot be normalized without modelx evaluation: {self.label}"
            ) from exc

    def _read_key(self, key: tuple[Any, ...]) -> Any:
        try:
            addr = self.mapping.get(key)
        except TypeError as exc:
            raise WholeArtifactError(
                f"unhashable Cell key cannot be a realized scheduled dependency: {self.label}{key!r}"
            ) from exc
        if addr is not None:
            if addr.pool == "double":
                if self.owner_checks[0] and int(self.downers[addr.offset]) != addr.node_id:
                    raise WholeArtifactError(
                        f"scheduled dependency requested before production: {self.label}{key!r}"
                    )
                return float(self.dslots[addr.offset])
            if addr.pool == "int64":
                if self.owner_checks[0] and int(self.iowners[addr.offset]) != addr.node_id:
                    raise WholeArtifactError(
                        f"scheduled dependency requested before production: {self.label}{key!r}"
                    )
                return int(self.islots[addr.offset])
            if addr.pool == "bool":
                if self.owner_checks[0] and int(self.bowners[addr.offset]) != addr.node_id:
                    raise WholeArtifactError(
                        f"scheduled dependency requested before production: {self.label}{key!r}"
                    )
                return bool(self.bslots[addr.offset])
            if addr.pool == "object":
                if self.owner_checks[0] and int(self.oowners[addr.offset]) != addr.node_id:
                    raise WholeArtifactError(
                        f"scheduled dependency requested before production: {self.label}{key!r}"
                    )
                return self.oslots[addr.offset]
            raise WholeArtifactError(f"unknown physical pool {addr.pool!r}")
        if key in self.input_values:
            return self.input_values[key]
        raise WholeArtifactError(
            f"Cell dependency is outside the realized sequential artifact: {self.label}{key!r}; "
            "recursive modelx fallback is disabled"
        )

    def match(self, *args, **kwargs):
        key = self._normalize(tuple(args), kwargs)
        keylen = len(key)
        Pair = namedtuple("ArgsValuePair", ["args", "value"])
        for match_len in range(keylen, -1, -1):
            for idxs in combinations(range(keylen), match_len):
                masked = [None] * keylen
                for idx in idxs:
                    masked[idx] = key[idx]
                tkey = tuple(masked)
                try:
                    value = self._read_key(tkey)
                except WholeArtifactError as exc:
                    # Exact realized matching only evaluates keys present in the
                    # captured graph. Missing masked keys mean the artifact lacks
                    # sufficient realized evidence; do not ask modelx to compute.
                    if "outside the realized sequential artifact" in str(exc):
                        continue
                    raise
                if value is not None:
                    return Pair(tkey, value)
        return Pair(None, None)

    def __call__(self, *args, **kwargs):
        raw = tuple(args)
        try:
            if not kwargs and raw in self.mapping:
                return self._read_key(raw)
        except TypeError:
            pass
        return self._read_key(self._normalize(raw, kwargs))

    def __getitem__(self, key):
        args = tuple(key) if isinstance(key, tuple) else (key,)
        try:
            if args in self.mapping:
                return self._read_key(args)
        except TypeError:
            pass
        return self._read_key(self._normalize(args, {}))


class _StrictSpaceProxy:
    __slots__ = ("factory", "original")

    def __init__(self, factory: "_StrictCallbackFactory", original: Any):
        self.factory = factory
        self.original = original

    def __getattr__(self, name):
        return self.factory.wrap_reference(getattr(self.original, name))

    def __getitem__(self, key):
        return self.factory.wrap_reference(self.original[key])

    def __call__(self, *args, **kwargs):
        return self.factory.wrap_reference(self.original(*args, **kwargs))

    def __iter__(self):
        return iter(self.original)

    def __len__(self):
        return len(self.original)


class _FrozenSpaceNamespace:
    """Direct zero-argument providers backed by compile-time captured objects.

    Missing attributes delegate to the strict graph proxy, whose resolver can only
    read scheduled slots or explicit inputs.  There is no modelx calculation path.
    """

    __slots__ = ("__dict__", "fallback")

    def __init__(self, providers: dict[str, Any], fallback: _StrictSpaceProxy):
        self.fallback = fallback
        for name, value in providers.items():
            # Bind through a default argument so every method retains its own value.
            self.__dict__[name] = (lambda _value=value: _value)

    def __getattr__(self, name):
        return getattr(self.fallback, name)

    def __getitem__(self, key):
        return self.fallback[key]

    def __call__(self, *args, **kwargs):
        return self.fallback(*args, **kwargs)


def _recursive_code_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names.update(_recursive_code_names(const))
    return names


class _StrictCallbackFactory:
    def __init__(
        self,
        compiler: RealizedTraceCompiler,
        typed: Any,
        dslots: np.ndarray,
        islots: np.ndarray,
        bslots: np.ndarray,
        oslots: list[Any],
        downers: np.ndarray,
        iowners: np.ndarray,
        bowners: np.ndarray,
        oowners: np.ndarray,
        frozen_space_providers: dict[int, dict[str, Any]] | None = None,
    ):
        # Runtime callbacks deliberately do not retain the RealizedTraceCompiler.
        # Space proxies may lazily resolve Cells while executing original formula
        # bytecode, so retain only the compact realized key -> physical-address
        # mapping required to construct strict resolvers on demand.
        self.node_addresses = typed.node_by_id
        self.dslots, self.islots, self.bslots, self.oslots = dslots, islots, bslots, oslots
        self.downers, self.iowners, self.bowners, self.oowners = downers, iowners, bowners, oowners
        # A mutable one-element flag is shared by every strict resolver. Checked
        # execution proves ownership; the production path disables those repeated
        # array reads after the exact sequential schedule has been proved.
        self.owner_checks = [True]
        self.obj_mappings: dict[int, dict[tuple[Any, ...], Any]] = {}
        for node in compiler.trace.nodes:
            addr = self.node_addresses.get(node.node_id)
            if addr is not None:
                self.obj_mappings.setdefault(id(node.obj), {})[node.args] = addr
        self._function_cache: dict[int, Any] = {}
        self._reference_cache: dict[int, Any] = {}
        self.frozen_space_providers = frozen_space_providers or {}

    def _resolver_for_impl(self, impl: Any, label: str) -> _StrictCellResolver:
        mapping = self.obj_mappings.get(id(impl), {})
        # Only explicit modelx input keys are allowed outside the scheduled graph.
        # Calculated cache entries are intentionally not snapshotted.
        input_values = {}
        for key in getattr(impl, "input_keys", ()):
            tkey = tuple(key) if isinstance(key, tuple) else (key,)
            try:
                input_values[tkey] = impl.data[tkey]
            except Exception:
                pass
        return _StrictCellResolver(
            mapping,
            self.dslots, self.islots, self.bslots, self.oslots,
            self.downers, self.iowners, self.bowners, self.oowners,
            self.owner_checks,
            input_values,
            label,
            impl,
        )

    def set_owner_checks(self, enabled: bool) -> None:
        self.owner_checks[0] = bool(enabled)

    def wrap_reference(self, value: Any):
        impl = _impl_from_global(value)
        if impl is not None:
            token = id(impl)
            cached = self._reference_cache.get(token)
            if cached is None:
                cached = self._resolver_for_impl(impl, getattr(impl, "fullname", getattr(impl, "name", "Cell")))
                self._reference_cache[token] = cached
            return cached
        if ((_MxBaseSpace and isinstance(value, _MxBaseSpace)) or (_MxNamespace and isinstance(value, _MxNamespace))):
            token = id(value)
            cached = self._reference_cache.get(token)
            if cached is None:
                strict = _StrictSpaceProxy(self, value)
                impl = getattr(value, "_impl", None)
                providers = self.frozen_space_providers.get(id(impl), {})
                cached = _FrozenSpaceNamespace(providers, strict) if providers else strict
                self._reference_cache[token] = cached
            return cached
        return value

    def callback_for_obj(self, obj: Any):
        token = id(obj)
        cached = self._function_cache.get(token)
        if cached is not None:
            return cached
        fn = obj.altfunc
        if any(ins.opname in {"STORE_GLOBAL", "DELETE_GLOBAL"} for ins in dis.get_instructions(fn)):
            raise WholeArtifactError(
                f"formula {getattr(obj, 'fullname', obj)!r} mutates globals; "
                "non-recursive artifact cannot safely clone this environment"
            )
        glb = dict(fn.__globals__)
        for name in _recursive_code_names(fn.__code__):
            if name in glb:
                glb[name] = self.wrap_reference(glb[name])
        cloned = types.FunctionType(fn.__code__, glb, fn.__name__, fn.__defaults__, fn.__closure__)
        cloned.__kwdefaults__ = getattr(fn, "__kwdefaults__", None)
        cloned.__annotations__ = getattr(fn, "__annotations__", {}).copy()
        self._function_cache[token] = cloned
        return cloned


_POOL_CODE = {"double": 0, "int64": 1, "bool": 2, "object": 3}
# Source-expanded variants remove the runtime variant/role interpreter.  The
# original 512-term budget forced a measured 703-term family back through that
# interpreter even though the generated Cython/C source remained modest.  A
# 1024-term budget is still bounded, keeps very large grammars table-driven, and
# is deliberately structural rather than tied to a domain model name.
_VARIANT_SOURCE_TERM_LIMIT = 1024
# A function-pointer role dispatcher avoids a large if/elif chain, but the
# indirect call itself costs more than the branch dispatcher on medium realized
# schedules.  Targeted profiling found the crossover between a ~40k-op family
# (RILA: no benefit) and a ~144k-op family (IUL: material benefit).  Keep this
# structural and conservative: specialize only when the realized family executes
# at least 100k role operations across its instances.
_DIRECT_ROLE_DISPATCH_MIN_OPERATIONS = 100_000


def _target_addresses(compiler: RealizedTraceCompiler, typed: Any) -> tuple[Any, ...]:
    runtime_to_id = compiler.trace.runtime_to_id
    out = []
    for obj, key in compiler.trace.target_runtime_nodes:
        nid = runtime_to_id.get((obj, tuple(key)))
        if nid is None or nid not in typed.node_by_id:
            raise WholeArtifactError("whole artifact target is not backed by a typed physical slot")
        out.append(typed.node_by_id[nid])
    return tuple(out)


def _read_address(addr: Any, dslots, islots, bslots, oslots):
    if addr.pool == "double":
        return float(dslots[addr.offset])
    if addr.pool == "int64":
        return int(islots[addr.offset])
    if addr.pool == "bool":
        return bool(bslots[addr.offset])
    if addr.pool == "object":
        return oslots[addr.offset]
    raise WholeArtifactError(f"unknown target pool {addr.pool!r}")


def _authoritative_target_values(compiler: RealizedTraceCompiler) -> tuple[Any, ...]:
    values = []
    for obj, key in compiler.trace.target_runtime_nodes:
        values.append(obj.get_value_from_key(tuple(key)))
    return tuple(values)


@dataclass
class WholeArtifactProgram:
    compiler: RealizedTraceCompiler | None
    typed: Any | None
    module: Any
    region_data: tuple[Any, ...]
    callback_factory: _StrictCallbackFactory
    binding_payloads: tuple[Any, ...]
    dslots: np.ndarray
    islots: np.ndarray
    bslots: np.ndarray
    oslots: list[Any]
    downers: np.ndarray
    iowners: np.ndarray
    bowners: np.ndarray
    oowners: np.ndarray
    targets: tuple[Any, ...]
    expected_targets: tuple[Any, ...]
    build_report: WholeArtifactBuildReport
    family_operations: int
    literal_operations: int
    register_region_failures: tuple[tuple[int, str], ...] = ()
    detached: bool = False
    proof_completed: bool = False

    @property
    def production_ready(self) -> bool:
        return bool(self.proof_completed or self.detached)

    def reset(self, *, production: bool = False) -> None:
        # The proved schedule overwrites every numeric value before use. Clearing
        # numeric slots and owner arrays on every production run is therefore pure
        # memory traffic. Object slots are still cleared to release references, and
        # dynamic binding cursors are always reset.
        if not production:
            self.dslots.fill(0.0)
            self.islots.fill(0)
            self.bslots.fill(0)
            self.downers.fill(-1)
            self.iowners.fill(-1)
            self.bowners.fill(-1)
            self.oowners.fill(-1)
        for i in range(len(self.oslots)):
            self.oslots[i] = None
        for payload in self.binding_payloads:
            payload.cursor.fill(0)

    def execute(self, *, validate: bool = True, prove_no_modelx: bool = True):
        if self.detached and (validate or prove_no_modelx):
            raise WholeArtifactError(
                "detached production artifact cannot validate against modelx or run proof instrumentation; "
                "use execute(validate=False, prove_no_modelx=False)"
            )
        if validate and not self.expected_targets:
            raise WholeArtifactError("validation reference values are unavailable")
        if prove_no_modelx and self.compiler is None:
            raise WholeArtifactError("modelx proof instrumentation is unavailable after production detachment")
        self.callback_factory.set_owner_checks(True)
        self.reset(production=False)
        events = None
        if prove_no_modelx:
            # This is proof instrumentation only, not part of artifact execution.
            # Clearing calculated caches makes an accidental evaluator escape
            # observable rather than letting a stale modelx cache hide it.
            assert self.compiler is not None
            self.compiler.sequential._clear_scheduled_values()
            system = self.compiler.trace.model._impl.system
            with system.trace_stack(maxlen=None):
                self.module.run_one(
                    self.dslots, self.islots, self.bslots, self.oslots,
                    self.downers, self.iowners, self.bowners, self.oowners,
                    self.region_data, True,
                )
                events = list(system.callstack.tracestack)
            if events:
                raise WholeArtifactError(
                    f"whole artifact triggered {len(events)} modelx trace events; recursive modelx execution is forbidden"
                )
        else:
            # Production/benchmark path. The generated driver and strict callback
            # environment contain the complete sequential schedule; no modelx
            # evaluator is entered here.
            self.module.run_one(
                self.dslots, self.islots, self.bslots, self.oslots,
                self.downers, self.iowners, self.bowners, self.oowners,
                self.region_data, True,
            )
        vals = tuple(_read_address(a, self.dslots, self.islots, self.bslots, self.oslots) for a in self.targets)
        result = vals[0] if len(vals) == 1 else vals
        exact = None
        max_abs = None
        if validate:
            exact = all(_hybrid_value_equal(a, e) for a, e in zip(vals, self.expected_targets))
            max_abs = _hybrid_numeric_error(list(vals), list(self.expected_targets))
            if not exact:
                raise WholeArtifactError(
                    f"non-recursive whole artifact target mismatch; max_abs_error={max_abs}"
                )
        if validate and prove_no_modelx and exact is True and not events:
            self.proof_completed = True
        stats = WholeArtifactExecutionStats(
            operation_count=self.build_report.operation_count,
            family_operations=self.family_operations,
            literal_operations=self.literal_operations,
            python_callback_operations=self.build_report.python_operations,
            native_operations=self.build_report.native_operations,
            modelx_trace_events=(len(events) if events is not None else None),
            exact=exact,
            max_abs_error=max_abs,
        )
        return (result, stats) if validate else result

    def runtime_payload(self) -> tuple[Any, ...]:
        """Return point-specific state consumed by generated ``run_one/run_batch``.

        This is intentionally an ABI payload, not a second semantic representation.
        The compiled module remains the canonical whole-artifact kernel; each point
        contributes only slots, provenance arrays and runtime region/binding data.
        """
        return (
            self.dslots, self.islots, self.bslots, self.oslots,
            self.downers, self.iowners, self.bowners, self.oowners,
            self.region_data,
        )

    def execute_production(self):
        """Execute the proved artifact without repeated ownership bookkeeping.

        Before proof completion this retains the established checked-write path,
        preserving compatibility for callers that deliberately build with
        ``prove=False``. Once proved (or detached), strict Cell resolvers read the
        same physical slots without owner-array checks and generated kernels omit
        owner writes.
        """
        if not self.production_ready:
            return self.execute(validate=False, prove_no_modelx=False)
        self.callback_factory.set_owner_checks(False)
        self.reset(production=True)
        self.module.run_one(
            self.dslots, self.islots, self.bslots, self.oslots,
            self.downers, self.iowners, self.bowners, self.oowners,
            self.region_data, False,
        )
        vals = tuple(
            _read_address(a, self.dslots, self.islots, self.bslots, self.oslots)
            for a in self.targets
        )
        return vals[0] if len(vals) == 1 else vals

    def detach_for_production(
        self, *, require_proof: bool = True, trim_allocator: bool = False
    ):
        """Drop trace/validation state after a successful exact non-recursion proof.

        Generated code, compact physical slots, runtime binding payloads and strict
        callback environments are retained. Calculated modelx caches are cleared
        before the compiler/trace metadata is released. This changes memory
        ownership only; execution semantics are unchanged.
        """
        if self.detached:
            return {
                "detached": True,
                "proof_completed": self.proof_completed,
                "allocator_trim_attempted": False,
                "allocator_trim_succeeded": False,
            }
        if require_proof and not self.proof_completed:
            raise WholeArtifactError(
                "production detachment requires a successful execute(validate=True, prove_no_modelx=True) proof"
            )
        compiler = self.compiler
        if compiler is None:
            raise WholeArtifactError("compiler metadata is already unavailable")

        self.reset()
        compiler.sequential._clear_scheduled_values()
        tracegraph_cleared = False
        try:
            compiler.trace.model._impl.tracegraph.clear()
            tracegraph_cleared = True
        except Exception:
            pass

        estimated_cache_bytes_released = 0
        if compiler.storage is not None:
            estimated_cache_bytes_released = int(
                compiler.storage.plan.baseline_shallow_value_bytes
            )

        # Drop heavyweight proof/planning state. Callback environments were built
        # above from compact realized address maps and do not retain this compiler.
        self.expected_targets = ()
        self.compiler = None
        self.typed = None
        self.detached = True

        import gc
        gc.collect()
        trim_succeeded = False
        if trim_allocator:
            try:
                import ctypes
                libc = ctypes.CDLL(None)
                malloc_trim = getattr(libc, "malloc_trim")
                malloc_trim.argtypes = [ctypes.c_size_t]
                malloc_trim.restype = ctypes.c_int
                trim_succeeded = bool(malloc_trim(0))
            except Exception:
                trim_succeeded = False

        return {
            "detached": True,
            "proof_completed": self.proof_completed,
            "tracegraph_cleared": tracegraph_cleared,
            "estimated_cache_bytes_released": estimated_cache_bytes_released,
            "allocator_trim_attempted": bool(trim_allocator),
            "allocator_trim_succeeded": trim_succeeded,
        }


@dataclass(frozen=True)
class _LiteralEmitSpec:
    index: int
    kind: str  # callback | prepared | native
    pool: str
    offset: int
    node_id: int
    function_name: str | None = None
    call_args: tuple[str, ...] = ()


def _literal_call_arg(value: Any, cython_type: str) -> str:
    if cython_type == "double" and isinstance(value, (int, float, np.integer, np.floating)):
        return repr(float(value))
    if cython_type == "long long" and isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return repr(int(value))
    if cython_type == "bint" and isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if cython_type == "str" and isinstance(value, str):
        return repr(value)
    if cython_type == "object" and (
        value is None or isinstance(value, (bool, int, float, str, np.bool_, np.integer, np.floating))
    ):
        if isinstance(value, np.generic):
            value = value.item()
        return repr(value)
    raise WholeArtifactError(f"literal native argument {value!r} does not match {cython_type}")


def _literal_store_lines(spec: _LiteralEmitSpec, value_expr: str, *, indent: str = "    ") -> list[str]:
    pool_arr = {"double": "dslots", "int64": "islots", "bool": "bslots", "object": "oslots"}[spec.pool]
    owner_arr = {"double": "downers", "int64": "iowners", "bool": "bowners", "object": "oowners"}[spec.pool]
    if spec.pool == "double":
        rhs = value_expr if spec.kind == "native" else f"<double>({value_expr})"
    elif spec.pool == "int64":
        rhs = value_expr if spec.kind == "native" else f"<long long>({value_expr})"
    elif spec.pool == "bool":
        rhs = f"1 if ({value_expr}) else 0"
    else:
        rhs = value_expr
    return [
        f"{indent}{pool_arr}[{spec.offset}] = {rhs}",
        f"{indent}if write_owners: {owner_arr}[{spec.offset}] = {spec.node_id}",
    ]


def _emit_literal_block_runner(block_index: int, specs: list[_LiteralEmitSpec]) -> list[str]:
    lines = [
        f"cdef void _run_literal_block_{block_index}(object data, double[::1] dslots, long long[::1] islots, unsigned char[::1] bslots, object oslots, long long[::1] downers, long long[::1] iowners, long long[::1] bowners, long long[::1] oowners, bint write_owners):",
        "    cdef object callbacks = data[0]",
        "    cdef object args = data[1]",
        "    cdef object prepared = data[6]",
        "    cdef object value",
    ]
    for spec in specs:
        if spec.kind == "prepared":
            lines.append(f"    value = prepared[{spec.index}]")
            lines.extend(_literal_store_lines(spec, "value"))
        elif spec.kind == "native":
            call = f"{spec.function_name}(" + ", ".join((*spec.call_args, "dslots", "islots", "bslots", "oslots")) + ")"
            lines.extend(_literal_store_lines(spec, call))
        else:
            lines.append(f"    value = callbacks[{spec.index}](*args[{spec.index}])")
            lines.extend(_literal_store_lines(spec, "value"))
    lines.append("")
    return lines


def _emit_literal_runner() -> list[str]:
    return [
        "cdef void _run_literal(object data, double[::1] dslots, long long[::1] islots, unsigned char[::1] bslots, object oslots, long long[::1] downers, long long[::1] iowners, long long[::1] bowners, long long[::1] oowners, bint write_owners):",
        "    cdef object callbacks = data[0]",
        "    cdef object args = data[1]",
        "    cdef long long[::1] pools = data[2]",
        "    cdef long long[::1] offsets = data[3]",
        "    cdef long long[::1] node_ids = data[4]",
        "    cdef long long[::1] modes = data[5]",
        "    cdef object prepared = data[6]",
        "    cdef Py_ssize_t j, n = len(callbacks), idx",
        "    cdef long long k, nid",
        "    cdef object value",
        "    for j in range(n):",
        "        if modes[j] == 1:",
        "            value = prepared[j]",
        "        else:",
        "            value = callbacks[j](*args[j])",
        "        k = pools[j]",
        "        idx = offsets[j]",
        "        nid = node_ids[j]",
        "        if k == 0:",
        "            dslots[idx] = <double>value",
        "            if write_owners: downers[idx] = nid",
        "        elif k == 1:",
        "            islots[idx] = <long long>value",
        "            if write_owners: iowners[idx] = nid",
        "        elif k == 2:",
        "            bslots[idx] = 1 if bool(value) else 0",
        "            if write_owners: bowners[idx] = nid",
        "        elif k == 3:",
        "            oslots[idx] = value",
        "            if write_owners: oowners[idx] = nid",
        "        else:",
        "            raise ValueError('unknown literal output pool')",
        "",
    ]


def _emit_family_context_type() -> list[str]:
    fields = [
        "double[::1] dslots", "long long[::1] islots", "unsigned char[::1] bslots", "object oslots",
        "long long[::1] downers", "long long[::1] iowners", "long long[::1] bowners", "long long[::1] oowners",
        "bint write_owners",
        "long long[::1] bind_kind", "long long[::1] bind_ia", "long long[::1] bind_ib", "double[::1] bind_da",
        "long long[::1] bind_payload_offset", "long long[::1] bind_payload_count",
        "long long[::1] bind_aux_offset", "long long[::1] bind_aux_count",
        "long long[::1] bind_i_payload", "double[::1] bind_d_payload", "object bind_object_payload",
        "long long[::1] bind_run_ends", "long long[::1] bind_cursor",
        "long long[::1] ref_kind", "long long[::1] ref_a", "long long[::1] ref_b", "long long[::1] ref_mod",
        "long long[::1] ref_lo", "long long[::1] ref_table_offset", "long long[::1] ref_table",
        "long long[::1] out_kind", "long long[::1] out_a", "long long[::1] out_b", "long long[::1] out_mod",
        "long long[::1] out_table_offset", "long long[::1] out_table",
    ]
    lines = ["cdef class _FamilyCtx:"]
    lines.extend(f"    cdef {field}" for field in fields)
    lines.append("")
    return lines


def _ctx_expr(expr: str) -> str:
    # Generated binding/output expressions use these canonical local names.
    # A Cython extension context keeps role helper signatures compact while all
    # accesses remain C-level cdef attributes.
    names = (
        "bind_payload_offset", "bind_payload_count", "bind_object_payload",
        "bind_aux_offset", "bind_aux_count", "bind_i_payload", "bind_d_payload",
        "bind_run_ends", "bind_cursor", "bind_kind", "bind_ia", "bind_ib", "bind_da",
        "out_table_offset", "out_table", "out_kind", "out_a", "out_b", "out_mod",
    )
    for name in names:
        expr = re.sub(rf"\b{re.escape(name)}\b", f"ctx.{name}", expr)
    return expr


def _emit_family_role_helpers(
    family_id: int,
    backend: Any,
    sites: tuple[Any, ...],
    typed: Any,
    compiler: Any,
    native_by_fid: dict[int, bool],
) -> list[str]:
    """Emit one role dispatcher per family, with every role body exactly once.

    Variant branches pass a literal role id. With optimization enabled the C
    compiler can inline/constant-fold this dispatch, while Cython sees one helper
    function per family rather than one function per role. This keeps translation
    cost tractable for phase-heavy families without expanding realized operations.
    """
    plan = compiler.structured.canonical_plan
    assert plan is not None
    output_kinds = _family_output_structural_kinds(compiler, typed, backend)
    site_by_role: dict[int, list[Any]] = {}
    for site in sites:
        site_by_role.setdefault(site.role, []).append(site)
    for rows in site_by_role.values():
        rows.sort(key=lambda x: x.arg_index)

    lines: list[str] = [
        f"cdef inline void _family_{family_id}_exec_role(Py_ssize_t role, Py_ssize_t occ, _FamilyCtx ctx, object callbacks, long long[::1] node_ids, long long[::1] role_base):",
        "    cdef Py_ssize_t idx, pos",
        "    cdef long long nid",
        "    cdef object py_value",
        "    cdef double d_value",
        "    cdef long long i_value",
        "    cdef bint b_value",
    ]
    formula_ids = backend.kernel.formula_op_ids
    for role, fid in enumerate(formula_ids):
        op = plan.formula_ops[fid]
        arg_sites = site_by_role.get(role, [])
        if len(arg_sites) != op.arity:
            raise WholeArtifactError(
                f"family {family_id} role {role} binding arity does not match FormulaOp"
            )
        arg_exprs = [_ctx_expr(_emit_runtime_arg_expr(site, "occ")) for site in arg_sites]
        prefix = "if" if role == 0 else "elif"
        lines.append(f"    {prefix} role == {role}:")
        lines.append(f"        pos = role_base[{role}] + occ")
        pool = typed.formula_by_id[fid].pool
        if native_by_fid.get(fid, False):
            callargs = ", ".join(
                ["occ"] + arg_exprs + [
                    "ctx.dslots", "ctx.islots", "ctx.bslots", "ctx.oslots",
                    "ctx.ref_kind", "ctx.ref_a", "ctx.ref_b", "ctx.ref_mod",
                    "ctx.ref_lo", "ctx.ref_table_offset", "ctx.ref_table",
                ]
            )
            call = f"f_{family_id}_{fid}({callargs})"
            target = {"double": "d_value", "int64": "i_value", "bool": "b_value", "object": "py_value"}.get(pool)
            if target is None:
                raise WholeArtifactError("unknown native FormulaOp output pool")
            lines.append(f"        {target} = {call}")
        else:
            args = ", ".join(arg_exprs)
            call = f"callbacks[pos]({args})" if args else "callbacks[pos]()"
            lines.append(f"        py_value = {call}")
        lines.append(f"        idx = {_ctx_expr(_emit_output_index_expr(role, 'occ', output_kinds[role]))}")
        lines.append("        nid = <long long>node_ids[pos]")
        if pool == "double":
            value = "d_value" if native_by_fid.get(fid, False) else "<double>py_value"
            lines += [f"        ctx.dslots[idx] = {value}", "        if ctx.write_owners: ctx.downers[idx] = nid"]
        elif pool == "int64":
            value = "i_value" if native_by_fid.get(fid, False) else "<long long>py_value"
            lines += [f"        ctx.islots[idx] = {value}", "        if ctx.write_owners: ctx.iowners[idx] = nid"]
        elif pool == "bool":
            value = "b_value" if native_by_fid.get(fid, False) else "(1 if bool(py_value) else 0)"
            lines += [f"        ctx.bslots[idx] = {value}", "        if ctx.write_owners: ctx.bowners[idx] = nid"]
        elif pool == "object":
            lines += ["        ctx.oslots[idx] = py_value", "        if ctx.write_owners: ctx.oowners[idx] = nid"]
        else:  # pragma: no cover
            raise WholeArtifactError(f"unknown role output pool {pool!r}")
        lines.append("        return")
    lines += [
        "    else:",
        "        raise ValueError('role id outside compiled family grammar')",
        "",
    ]
    return lines


def _emit_family_direct_role_functions(
    family_id: int,
    backend: Any,
    sites: tuple[Any, ...],
    typed: Any,
    compiler: Any,
    native_by_fid: dict[int, bool],
) -> list[str]:
    """Emit one direct C function per role plus a C function-pointer table.

    Large grammars use a runtime variant program to keep generated source bounded.
    Routing every realized operation through one 50-100+ arm ``if/elif`` role
    dispatcher becomes a measurable native scheduling tax.  Splitting the exact
    same role bodies into C functions keeps one canonical family/runtime ABI while
    turning dynamic role selection into one indexed C function-pointer call.
    """
    plan = compiler.structured.canonical_plan
    assert plan is not None
    output_kinds = _family_output_structural_kinds(compiler, typed, backend)
    site_by_role: dict[int, list[Any]] = {}
    for site in sites:
        site_by_role.setdefault(site.role, []).append(site)
    for rows in site_by_role.values():
        rows.sort(key=lambda x: x.arg_index)

    formula_ids = backend.kernel.formula_op_ids
    lines: list[str] = []
    for role, fid in enumerate(formula_ids):
        op = plan.formula_ops[fid]
        arg_sites = site_by_role.get(role, [])
        if len(arg_sites) != op.arity:
            raise WholeArtifactError(
                f"family {family_id} role {role} binding arity does not match FormulaOp"
            )
        arg_exprs = [_ctx_expr(_emit_runtime_arg_expr(site, "occ")) for site in arg_sites]
        lines += [
            f"cdef void _family_{family_id}_role_{role}(Py_ssize_t occ, _FamilyCtx ctx, object callbacks, long long[::1] node_ids, long long[::1] role_base) except *:",
            "    cdef Py_ssize_t idx, pos",
            "    cdef long long nid",
            "    cdef object py_value",
            "    cdef double d_value",
            "    cdef long long i_value",
            "    cdef bint b_value",
            f"    pos = role_base[{role}] + occ",
        ]
        pool = typed.formula_by_id[fid].pool
        if native_by_fid.get(fid, False):
            callargs = ", ".join(
                ["occ"] + arg_exprs + [
                    "ctx.dslots", "ctx.islots", "ctx.bslots", "ctx.oslots",
                    "ctx.ref_kind", "ctx.ref_a", "ctx.ref_b", "ctx.ref_mod",
                    "ctx.ref_lo", "ctx.ref_table_offset", "ctx.ref_table",
                ]
            )
            call = f"f_{family_id}_{fid}({callargs})"
            target = {"double": "d_value", "int64": "i_value", "bool": "b_value", "object": "py_value"}.get(pool)
            if target is None:
                raise WholeArtifactError("unknown native FormulaOp output pool")
            lines.append(f"    {target} = {call}")
        else:
            args = ", ".join(arg_exprs)
            call = f"callbacks[pos]({args})" if args else "callbacks[pos]()"
            lines.append(f"    py_value = {call}")
        lines.append(f"    idx = {_ctx_expr(_emit_output_index_expr(role, 'occ', output_kinds[role]))}")
        lines.append("    nid = <long long>node_ids[pos]")
        if pool == "double":
            value = "d_value" if native_by_fid.get(fid, False) else "<double>py_value"
            lines += [f"    ctx.dslots[idx] = {value}", "    if ctx.write_owners: ctx.downers[idx] = nid"]
        elif pool == "int64":
            value = "i_value" if native_by_fid.get(fid, False) else "<long long>py_value"
            lines += [f"    ctx.islots[idx] = {value}", "    if ctx.write_owners: ctx.iowners[idx] = nid"]
        elif pool == "bool":
            value = "b_value" if native_by_fid.get(fid, False) else "(1 if bool(py_value) else 0)"
            lines += [f"    ctx.bslots[idx] = {value}", "    if ctx.write_owners: ctx.bowners[idx] = nid"]
        elif pool == "object":
            lines += ["    ctx.oslots[idx] = py_value", "    if ctx.write_owners: ctx.oowners[idx] = nid"]
        else:  # pragma: no cover
            raise WholeArtifactError(f"unknown role output pool {pool!r}")
        lines.append("")

    role_count = int(backend.kernel.role_count)
    lines += [
        f"ctypedef void (*_family_{family_id}_role_fn_t)(Py_ssize_t, _FamilyCtx, object, long long[::1], long long[::1]) except *",
        f"cdef _family_{family_id}_role_fn_t _family_{family_id}_role_fns[{role_count}]",
        f"cdef bint _family_{family_id}_role_fns_ready = False",
        f"cdef inline void _family_{family_id}_init_role_fns():",
        f"    global _family_{family_id}_role_fns_ready",
        f"    if _family_{family_id}_role_fns_ready: return",
    ]
    for role in range(role_count):
        lines.append(f"    _family_{family_id}_role_fns[{role}] = _family_{family_id}_role_{role}")
    lines += [f"    _family_{family_id}_role_fns_ready = True", ""]
    return lines


def _runtime_variant_program(backend: Any) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    roles: list[int] = []
    for pattern in backend.kernel.variants:
        roles.extend(int(role) for role in pattern)
        offsets.append(len(roles))
    return np.asarray(offsets, dtype=np.int64), np.asarray(roles, dtype=np.int64)


def _variant_source_term_count(backend: Any) -> int:
    return sum(len(pattern) for pattern in backend.kernel.variants)


def _use_runtime_variant_program(backend: Any) -> bool:
    return _variant_source_term_count(backend) > _VARIANT_SOURCE_TERM_LIMIT


def _realized_family_operation_count(backend: Any) -> int:
    variant_lengths = tuple(len(pattern) for pattern in backend.kernel.variants)
    total = 0
    for instance in backend.instances:
        total += sum(variant_lengths[int(variant)] for variant in instance.variant_ids)
    return int(total)


def _use_direct_role_dispatch(backend: Any) -> bool:
    return (
        _use_runtime_variant_program(backend)
        and _realized_family_operation_count(backend) >= _DIRECT_ROLE_DISPATCH_MIN_OPERATIONS
    )


def _emit_family_runner(
    family_id: int,
    backend: Any,
    sites: tuple[Any, ...],
    typed: Any,
    compiler: Any,
    native_by_fid: dict[int, bool],
) -> list[str]:
    dynamic_program = _use_runtime_variant_program(backend)
    direct_role_dispatch = _use_direct_role_dispatch(backend)
    lines = [
        f"cdef void _family_{family_id}(object data, double[::1] dslots, long long[::1] islots, unsigned char[::1] bslots, object oslots, long long[::1] downers, long long[::1] iowners, long long[::1] bowners, long long[::1] oowners, bint write_owners):",
        "    cdef long long[::1] variants = data[0]",
        "    cdef object callbacks = data[27]",
        "    cdef long long[::1] node_ids = data[28]",
        "    cdef long long[::1] role_base = data[29]",
        "    cdef _FamilyCtx ctx = _FamilyCtx()",
        "    cdef Py_ssize_t iteration, variant, role, occ, p, start, stop",
        "    ctx.dslots = dslots; ctx.islots = islots; ctx.bslots = bslots; ctx.oslots = oslots",
        "    ctx.downers = downers; ctx.iowners = iowners; ctx.bowners = bowners; ctx.oowners = oowners",
        "    ctx.write_owners = write_owners",
        "    ctx.bind_kind = data[1]; ctx.bind_ia = data[2]; ctx.bind_ib = data[3]; ctx.bind_da = data[4]",
        "    ctx.bind_payload_offset = data[5]; ctx.bind_payload_count = data[6]",
        "    ctx.bind_aux_offset = data[7]; ctx.bind_aux_count = data[8]",
        "    ctx.bind_i_payload = data[9]; ctx.bind_d_payload = data[10]; ctx.bind_object_payload = data[11]",
        "    ctx.bind_run_ends = data[12]; ctx.bind_cursor = data[13]",
        "    ctx.ref_kind = data[14]; ctx.ref_a = data[15]; ctx.ref_b = data[16]; ctx.ref_mod = data[17]",
        "    ctx.ref_lo = data[18]; ctx.ref_table_offset = data[19]; ctx.ref_table = data[20]",
        "    ctx.out_kind = data[21]; ctx.out_a = data[22]; ctx.out_b = data[23]; ctx.out_mod = data[24]",
        "    ctx.out_table_offset = data[25]; ctx.out_table = data[26]",
    ]
    if dynamic_program:
        lines += [
            "    cdef long long[::1] variant_offsets = data[30]",
            "    cdef long long[::1] variant_roles = data[31]",
            "    cdef long long[::1] counters = data[32]",
        ]
        if direct_role_dispatch:
            lines.append(f"    _family_{family_id}_init_role_fns()")
        lines += [
            f"    for role in range({backend.kernel.role_count}): counters[role] = 0",
            "    for iteration in range(variants.shape[0]):",
            "        variant = variants[iteration]",
            "        if variant < 0 or variant + 1 >= variant_offsets.shape[0]: raise ValueError('variant id outside compiled family grammar')",
            "        start = variant_offsets[variant]",
            "        stop = variant_offsets[variant + 1]",
            "        for p in range(start, stop):",
            "            role = variant_roles[p]",
            "            occ = counters[role]",
        ]
        if direct_role_dispatch:
            lines.append(f"            _family_{family_id}_role_fns[role](occ, ctx, callbacks, node_ids, role_base)")
        else:
            lines.append(f"            _family_{family_id}_exec_role(role, occ, ctx, callbacks, node_ids, role_base)")
        lines += [
            "            counters[role] = occ + 1",
        ]
        for role in range(backend.kernel.role_count):
            lines.append(
                f"    if counters[{role}] != role_base[{role + 1}] - role_base[{role}]: raise ValueError('family role occurrence count mismatch')"
            )
    else:
        for role in range(backend.kernel.role_count):
            lines.append(f"    cdef Py_ssize_t c_{role} = 0")
        lines += [
            "    for iteration in range(variants.shape[0]):",
            "        variant = variants[iteration]",
        ]
        for variant_id, pattern in enumerate(backend.kernel.variants):
            prefix = "if" if variant_id == 0 else "elif"
            lines.append(f"        {prefix} variant == {variant_id}:")
            if not pattern:
                lines.append("            pass")
                continue
            for role in pattern:
                lines.append(
                    f"            _family_{family_id}_exec_role({role}, c_{role}, ctx, callbacks, node_ids, role_base)"
                )
                lines.append(f"            c_{role} += 1")
        lines += [
            "        else:",
            "            raise ValueError('variant id outside compiled family grammar')",
        ]
        for role in range(backend.kernel.role_count):
            lines.append(
                f"    if c_{role} != role_base[{role + 1}] - role_base[{role}]: raise ValueError('family role occurrence count mismatch')"
            )
    lines.append("")
    return lines


def _native_scalar_environment_stable(compiler: RealizedTraceCompiler, formula_op_id: int, nodes: list[Any] | None = None) -> bool:
    """Reject shared native code when an inlined scalar global varies.

    Runtime Cell addresses are descriptor-driven, but ``_RuntimeFormulaEmitter``
    still emits ordinary numeric globals as C literals.  A FormulaOp may retain
    one code identity while concrete itemspaces carry different globals.  Native
    reuse is therefore valid only when every scalar global visible to the formula
    has the same value across all realized occurrences.  Non-scalar globals are
    handled by the normal native-capability/reference gates.
    """
    plan = compiler.structured.canonical_plan
    assert plan is not None
    op = plan.formula_ops[formula_op_id]
    if nodes is None:
        nodes = [n for n in compiler.trace.nodes if n.shape_token == op.role]
    if not nodes:
        return False
    rep_fn = nodes[0].obj.altfunc
    names = _recursive_code_names(rep_fn.__code__)

    def scalar_signature(value: Any):
        if isinstance(value, (bool, np.bool_)):
            return ("bool", bool(value))
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return ("int", int(value))
        if isinstance(value, (float, np.floating)):
            value = float(value)
            if np.isnan(value):
                return ("float_nan",)
            # repr preserves the exact Python float used by the emitter.
            return ("float", repr(value))
        if isinstance(value, str):
            return ("str", value)
        return None

    rep_globals = rep_fn.__globals__
    scalar_names = {
        name: scalar_signature(rep_globals[name])
        for name in names
        if name in rep_globals and scalar_signature(rep_globals[name]) is not None
    }
    for node in nodes[1:]:
        glb = node.obj.altfunc.__globals__
        for name, expected in scalar_names.items():
            if name not in glb or scalar_signature(glb[name]) != expected:
                return False
    return True


def _engine_object_globals_for_formula(
    compiler: RealizedTraceCompiler, formula_op_id: int, formula: Any
) -> tuple[dict[str, Any], str | None]:
    plan = compiler.structured.canonical_plan
    assert plan is not None
    op = plan.formula_ops[formula_op_id]
    nodes = [n for n in compiler.trace.nodes if n.shape_token == op.role]
    if not nodes:
        return {}, "no_representative_nodes"
    formals = {a.arg for a in formula.function.args.args}
    # Decorators and Cython annotations belong to the translated source ABI, not
    # to the formula's runtime environment.  Walking the complete FunctionDef
    # incorrectly classified names such as ``_mx_cy``, ``object`` and ``str``
    # as model globals and rejected every translated formula.
    body_nodes = [child for stmt in formula.function.body for child in ast.walk(stmt)]
    locals_: set[str] = set()

    def add_local_target(target: ast.AST):
        if isinstance(target, ast.Name):
            locals_.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                add_local_target(item)

    for node in body_nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                add_local_target(target)
        if isinstance(node, ast.NamedExpr):
            add_local_target(node.target)
        if isinstance(node, (ast.comprehension, ast.For)):
            add_local_target(node.target)
        if isinstance(node, ast.Lambda):
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                locals_.add(arg.arg)
        if isinstance(node, ast.ExceptHandler) and node.name:
            locals_.add(node.name)
    def frozen_io_snapshot(value: Any) -> tuple[bool, Any]:
        """Detach supported immutable modelx IO interfaces from their manager."""
        module_name = type(value).__module__
        class_name = type(value).__name__
        if module_name == "modelx.io.excelio" and class_name == "ExcelRange":
            return True, dict(value.items())
        if module_name == "modelx.io.pandasio" and class_name == "PandasData":
            payload = value.value
            if type(payload).__module__.startswith("pandas.") and hasattr(payload, "copy"):
                payload = payload.copy(deep=True)
            return True, payload
        return False, None

    qualified_cell_bases: set[int] = set()
    qualified_io_snapshots: dict[str, Any] = {}
    for child in body_nodes:
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        if isinstance(child.func.value, ast.Name):
            base_node = child.func.value
        elif (
            isinstance(child.func.value, ast.Subscript)
            and isinstance(child.func.value.value, ast.Name)
        ):
            base_node = child.func.value.value
        else:
            continue
        base_name = base_node.id
        attr_name = child.func.attr
        call_name = _cell_call_name(child)
        if call_name is None:
            continue
        resolved = []
        io_values = []
        for concrete in nodes:
            globals_ = concrete.obj.altfunc.__globals__
            if base_name not in globals_:
                resolved = []
                break
            impl = _reference_impl_from_formula_global(globals_, call_name, child)
            if impl is not None:
                resolved.append(impl)
                continue
            # Only the direct ``space.io_ref()`` shape can denote a frozen IO
            # payload. Parameterized Space selection remains a Cell-only path.
            if not isinstance(child.func.value, ast.Name):
                resolved = []
                io_values = []
                break
            try:
                value = getattr(globals_[base_name], attr_name)
            except (AttributeError, KeyError, TypeError):
                resolved = []
                break
            is_io, snapshot = frozen_io_snapshot(value)
            if not is_io or child.args or child.keywords:
                resolved = []
                io_values = []
                break
            io_values.append((value, snapshot))
        if len(resolved) == len(nodes):
            qualified_cell_bases.add(id(base_node))
        elif len(io_values) == len(nodes):
            first_value = io_values[0][0]
            if all(value is first_value for value, _snapshot in io_values[1:]):
                qualified_cell_bases.add(id(base_node))
                qualified_io_snapshots[f"{base_name}.{attr_name}"] = io_values[0][1]

    used = {
        node.id
        for node in body_nodes
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and id(node) not in qualified_cell_bases
    }
    result: dict[str, Any] = dict(qualified_io_snapshots)
    builtin_names = {
        "abs", "min", "max", "float", "int", "bool", "str", "object",
        "round", "range", "len", "sum", "list", "tuple", "sorted", "hasattr",
        "ValueError", "TypeError", "RuntimeError", "True", "False", "None",
    }
    for name in sorted(used - formals - locals_ - builtin_names):
        values = []
        is_cell = False
        for node in nodes:
            glb = node.obj.altfunc.__globals__
            if name not in glb:
                return {}, f"missing_global:{name}"
            value = glb[name]
            if _impl_from_global(value) is not None:
                is_cell = True
                break
            values.append(value)
        if is_cell:
            continue
        first = values[0]
        module_name = type(first).__module__
        if module_name == "modelx" or module_name.startswith("modelx."):
            is_io, snapshot = frozen_io_snapshot(first)
            if is_io:
                if any(value is not first for value in values[1:]):
                    return {}, f"object_global_changes:{name}"
                result[name] = snapshot
                continue
            # Keeping a modelx Space/Interface object in an otherwise detached
            # Cython function silently reintroduces model recursion.  A later
            # frozen-provider pass can replace zero-argument attribute calls;
            # until then this formula remains an explicit Python boundary.
            return {}, f"modelx_object_global:{name}"
        if isinstance(first, (bool, np.bool_, int, np.integer, float, np.floating, str)) or first is None:
            continue
        def signature(value):
            impl = getattr(value, "_impl", None)
            return (type(value), id(impl) if impl is not None else id(value))
        expected = signature(first)
        if any(signature(value) != expected for value in values[1:]):
            return {}, f"object_global_changes:{name}"
        result[name] = first
    return result, None


def _frozen_static_object_formula_values(
    compiler: RealizedTraceCompiler, typed: Any
) -> dict[int, Any]:
    """Prepare zero-argument shared object Cells as immutable artifact inputs.

    These values are model data, not projection state. Re-running a provider
    formula such as ``read_csv`` on every artifact invocation would discard a
    useful part of modelx-cython's execution model. The trace has already
    produced the exact object, so the generated graph kernel can read it from a
    prepared global without invoking modelx or its cache.

    The rule is fail-closed: every realized node for the FormulaOp must be a
    zero-argument Cell on a non-ItemSpace, and all nodes must contain the same
    object identity. Dynamic parameterized-space values are never frozen here.
    """
    plan = compiler.structured.canonical_plan
    assert plan is not None
    nodes_by_role: dict[Any, list[Any]] = {}
    for node in compiler.trace.nodes:
        nodes_by_role.setdefault(node.shape_token, []).append(node)
    result: dict[int, Any] = {}
    for fid, output in typed.formula_by_id.items():
        if output.pool != "object":
            continue
        nodes = nodes_by_role.get(plan.formula_ops[fid].role, ())
        if not nodes:
            continue
        values: list[Any] = []
        for node in nodes:
            parent = getattr(node.obj, "parent", None)
            if tuple(node.args) or parent is None or parent.__class__.__name__ == "ItemSpaceImpl":
                values = []
                break
            try:
                values.append(node.obj.data[tuple(node.args)])
            except Exception:
                values = []
                break
        if not values:
            continue
        first = values[0]
        if any(value is not first for value in values[1:]):
            continue
        result[fid] = first
    return result


def _emit_frozen_static_object_function(
    function_name: str, arg_types: list[tuple[str, str]], global_index: int
) -> list[str]:
    args = ", ".join(f"{typ} {name}" for name, typ in arg_types)
    if args:
        args += ", "
    return [
        f"cdef inline object {function_name}(Py_ssize_t formula_occ, {args}"
        "double[::1] dslots, long long[::1] islots, unsigned char[::1] bslots, object oslots, "
        "long long[::1] ref_kind, long long[::1] ref_a, long long[::1] ref_b, "
        "long long[::1] ref_mod, long long[::1] ref_lo, "
        "long long[::1] ref_table_offset, long long[::1] ref_table):",
        f"    return _mxg_engine_globals[{int(global_index)}]",
    ]


def _emit_engine_global_support() -> list[str]:
    return [
        "cdef object _mxg_engine_globals = ()",
        "cpdef _init_engine_globals(object payload):",
        "    global _mxg_engine_globals",
        "    _mxg_engine_globals = payload",
        "    return None",
        "",
    ]


def _prepare_frozen_object_providers(compiler: RealizedTraceCompiler, typed: Any) -> dict[int, dict[str, Any]]:
    """Capture realized zero-argument object providers once, outside runtime.

    This is deliberately structural: any non-dynamic Space with a zero-argument
    object Cell in the realized graph can provide immutable/shared input objects.
    Dynamic ItemSpaces are excluded because their values are point state.
    """
    result: dict[int, dict[str, Any]] = {}
    for node in compiler.trace.nodes:
        addr = typed.node_by_id.get(node.node_id)
        if addr is None or addr.pool != "object" or tuple(node.args):
            continue
        parent = getattr(node.obj, "parent", None)
        if parent is None or parent.__class__.__name__ == "ItemSpaceImpl":
            continue
        name = getattr(node.obj, "name", None)
        if not isinstance(name, str) or not name:
            continue
        try:
            value = node.obj.data[tuple(node.args)]
        except Exception:
            continue
        result.setdefault(id(parent), {})[name] = value
    return result


def _value_kind(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray:{value.dtype}:shape={tuple(int(x) for x in value.shape)}"
    shape = getattr(value, "shape", None)
    module = type(value).__module__.split(".", 1)[0]
    name = type(value).__name__
    if module == "pandas" and shape is not None:
        return f"pandas.{name}:shape={tuple(int(x) for x in shape)}"
    if isinstance(value, np.generic):
        return f"numpy_scalar:{value.dtype}"
    return f"scalar:{type(value).__module__}.{name}"


def build_whole_artifact_cython(
    compiler: RealizedTraceCompiler,
    build_dir: str | Path,
    *,
    module_prefix: str = "mxg_whole_artifact",
    optimization: str = "O1",
    experimental_allow_pow: bool = False,
    experimental_register_regions: bool = False,
    experimental_register_region_blocks: tuple[int, ...] | None = None,
    experimental_frozen_references: bool = False,
    execution_profile: str = "manual",
    modelx_cython_formula_catalog: ModelxCythonFormulaCatalog | None = None,
    modelx_cython_formula_package: str | Path | None = None,
    experimental_literal_lowering: bool = True,
) -> WholeArtifactProgram:
    """Build a single-entry, non-recursive sequential artifact.

    This foundation deliberately executes FormulaOps as original Python bytecode
    callbacks while moving *all scheduling* into one generated Cython artifact.
    Recovered family control executes as exact variant loops; literal regions use
    compact runtime callback schedules. Every scheduled Cell dependency is rebound
    to compiler-owned physical slots and modelx recursive evaluation is forbidden.

    When ``experimental_register_regions`` is enabled, proven numeric RegisterRegion
    blocks are composed as internal Cython calls inside this same module. Unsupported
    regions retain the established family runner; there is never a Python call per
    optimized region.
    """
    if compiler.structured is None:
        compiler.recover_loops()
    if compiler.slots is None:
        compiler.optimize_storage(); compiler.lower_slots()
    analysis = compiler.native_plan or compiler.analyze_native()
    typed = _typed_slot_plan(compiler, analysis, include_objects=True)
    plan = compiler.structured.canonical_plan
    if plan is None:
        raise WholeArtifactError("canonical execution plan is required")

    if modelx_cython_formula_catalog is not None and modelx_cython_formula_package is not None:
        raise WholeArtifactError(
            "pass either modelx_cython_formula_catalog or modelx_cython_formula_package, not both"
        )
    if modelx_cython_formula_package is not None:
        modelx_cython_formula_catalog = ModelxCythonFormulaCatalog.from_package(
            modelx_cython_formula_package
        )

    # Production formula compilation is self-contained. The realized trace and
    # typed slot plan contain the source/ABI evidence needed to emit ordinary
    # formulas as Cython functions. A modelx-cython translation is therefore an
    # optional comparison oracle, not a build or runtime dependency.
    if modelx_cython_formula_catalog is None:
        modelx_cython_formula_catalog = ModelxCythonFormulaCatalog.from_compiler(
            compiler, typed
        )

    engine_formula_by_fid: dict[int, Any] = {}
    engine_rejections: dict[int, str] = {}
    if modelx_cython_formula_catalog is not None:
        for op in plan.formula_ops:
            formula = modelx_cython_formula_catalog.match_formula_op(op.op_id)
            if formula is None:
                parts = str(op.fullname).split(".")
                if len(parts) < 2:
                    engine_rejections[op.op_id] = "fullname_has_no_space"
                    continue
                formula = modelx_cython_formula_catalog.match(parts[-2], parts[-1])
            if formula is None:
                engine_rejections[op.op_id] = "no_unique_translated_formula"
                continue
            engine_formula_by_fid[op.op_id] = formula

    engine_catalog_matched_fids = set(engine_formula_by_fid)

    engine_reference_pool_hints: dict[tuple[int, str], str] = {}
    if modelx_cython_formula_catalog is not None:
        abi_pool = {"double": "double", "long long": "int64", "bint": "bool", "str": "object", "object": "object"}
        for fid, formula in engine_formula_by_fid.items():
            for call in ast.walk(formula.function):
                if not isinstance(call, ast.Call):
                    continue
                name = _cell_call_name(call)
                if name is None:
                    continue
                cell_name = name.rsplit(".", 1)[-1]
                target_formula = modelx_cython_formula_catalog.match(formula.space_name, cell_name)
                if target_formula is None:
                    candidates = modelx_cython_formula_catalog.by_cell_name(cell_name)
                    signatures = {x.return_type for x in candidates}
                    target_formula = candidates[0] if candidates and len(signatures) == 1 else None
                if target_formula is not None and target_formula.return_type in abi_pool:
                    engine_reference_pool_hints[(fid, name)] = abi_pool[target_formula.return_type]

    frozen_static_object_values = _frozen_static_object_formula_values(compiler, typed)
    # These provider FormulaOps are prepared once and do not execute their
    # translated file-loading/object-construction bodies at runtime.
    for fid in frozen_static_object_values:
        engine_formula_by_fid.pop(fid, None)

    engine_object_global_indices: dict[int, dict[str, int]] = {}
    engine_object_global_values: list[Any] = []
    frozen_static_object_indices: dict[int, int] = {}
    for fid, value in sorted(frozen_static_object_values.items()):
        frozen_static_object_indices[fid] = len(engine_object_global_values)
        engine_object_global_values.append(value)
    for fid, formula in list(engine_formula_by_fid.items()):
        values, reason = _engine_object_globals_for_formula(compiler, fid, formula)
        if reason is not None:
            engine_rejections[fid] = reason
            engine_formula_by_fid.pop(fid, None)
            continue
        indices: dict[str, int] = {}
        for name, value in values.items():
            indices[name] = len(engine_object_global_values)
            engine_object_global_values.append(value)
        engine_object_global_indices[fid] = indices

    frozen_plan = compiler.frozen_reference_plan if experimental_frozen_references else None
    if experimental_frozen_references and frozen_plan is None:
        frozen_plan = compiler.plan_frozen_numeric_references()

    expected_targets = _authoritative_target_values(compiler)
    targets = _target_addresses(compiler, typed)

    dslots = np.zeros(typed.double_count, dtype=np.float64)
    islots = np.zeros(typed.int_count, dtype=np.int64)
    bslots = np.zeros(typed.bool_count, dtype=np.uint8)
    oslots: list[Any] = [None] * typed.object_count
    downers = np.full(typed.double_count, -1, dtype=np.int64)
    iowners = np.full(typed.int_count, -1, dtype=np.int64)
    bowners = np.full(typed.bool_count, -1, dtype=np.int64)
    oowners = np.full(typed.object_count, -1, dtype=np.int64)
    frozen_space_providers = _prepare_frozen_object_providers(compiler, typed)
    callbacks = _StrictCallbackFactory(
        compiler, typed, dslots, islots, bslots, oslots,
        downers, iowners, bowners, oowners,
        frozen_space_providers=frozen_space_providers,
    )

    analysis_by_id = {row.formula_op_id: row for row in analysis.formulae}
    nodes_by_role: dict[Any, list[Any]] = {}
    for node in compiler.trace.nodes:
        nodes_by_role.setdefault(node.shape_token, []).append(node)
    family_backend: dict[int, Any] = {}
    family_sites: dict[int, tuple[Any, ...]] = {}
    family_native: dict[int, dict[int, bool]] = {}
    family_frozen: dict[int, dict[int, bool]] = {}
    family_engine: dict[int, dict[int, Any]] = {}
    family_static_objects: dict[int, dict[int, int]] = {}
    family_references: dict[int, Any] = {}
    family_ordinal_by_block: dict[tuple[int, int], int] = {}
    for family in plan.code_families:
        backend = build_native_family_backend_plan(compiler, family.family_id)
        sites = build_runtime_arg_sites(backend)
        native = {
            fid: (
                _exact_native_formula_ok(
                    compiler, fid, analysis_by_id[fid], allow_pow=experimental_allow_pow
                )
                and _native_scalar_environment_stable(
                    compiler, fid, nodes_by_role.get(plan.formula_ops[fid].role, [])
                )
            )
            for fid in set(backend.kernel.formula_op_ids)
        }
        engine_for_family: dict[int, Any] = {}
        static_objects_for_family: dict[int, int] = {}
        for fid in set(backend.kernel.formula_op_ids):
            if fid in frozen_static_object_indices:
                native[fid] = True
                static_objects_for_family[fid] = frozen_static_object_indices[fid]
                continue
            formula = engine_formula_by_fid.get(fid)
            output = typed.formula_by_id.get(fid)
            if formula is None or output is None:
                continue
            if not modelx_cython_formula_native_ok(formula, output.pool):
                engine_rejections[fid] = f"translated_return_abi:{formula.return_type}:pool={output.pool}"
                continue
            if not _native_scalar_environment_stable(
                compiler, fid, nodes_by_role.get(plan.formula_ops[fid].role, [])
            ):
                engine_rejections[fid] = "scalar_environment_changes"
                continue
            native[fid] = True
            engine_for_family[fid] = formula
        frozen_candidates: set[int] = set()
        frozen_native = {fid: False for fid in set(backend.kernel.formula_op_ids)}
        if frozen_plan is not None and frozen_plan.site_count:
            from .frozen_references import frozen_formula_ast_exact_ok
            for fid in set(backend.kernel.formula_op_ids):
                if not frozen_plan.sites_for_formula(fid):
                    continue
                if frozen_formula_ast_exact_ok(
                    compiler, fid, analysis_by_id[fid], allow_pow=experimental_allow_pow
                ) and _native_scalar_environment_stable(
                    compiler, fid, nodes_by_role.get(plan.formula_ops[fid].role, [])
                ):
                    native[fid] = True
                    frozen_candidates.add(fid)
        for role, fid in enumerate(backend.kernel.formula_op_ids):
            if any(site.cython_type == "object" for site in sites if site.role == role):
                # modelx-cython's traced ABI can legitimately accept strings or
                # Python objects. The legacy scalar emitter cannot.
                if fid not in engine_for_family:
                    native[fid] = False
        reference_native = {
            fid: ok and fid not in static_objects_for_family
            for fid, ok in native.items()
        }
        refs = build_reference_binding_plan(
            compiler, typed, backend, reference_native,
            formula_functions={fid: f.function for fid, f in engine_for_family.items()},
            occurrence_formula_ops=set(engine_for_family),
            reference_pool_hints=engine_reference_pool_hints,
        )
        for fid in refs.unsupported_formula_ops:
            native[fid] = False
            if fid in engine_for_family:
                reason = next((r for ff, r in reversed(refs.fallback_reasons) if ff == fid), "reference_plan")
                engine_rejections[fid] = reason
        sites_by_role: dict[int, list[Any]] = {}
        for site in sites:
            sites_by_role.setdefault(site.role, []).append(site)
        for rows in sites_by_role.values():
            rows.sort(key=lambda x: x.arg_index)
        for fid, formula in list(engine_for_family.items()):
            if not native.get(fid, False):
                engine_for_family.pop(fid, None)
                continue
            try:
                for role, ffid in enumerate(backend.kernel.formula_op_ids):
                    if ffid != fid:
                        continue
                    rows = sites_by_role.get(role, [])
                    formal_names = [a.arg for a in formula.function.args.args]
                    if len(formal_names) != len(rows) or len(formal_names) != len(formula.arg_types):
                        raise WholeArtifactError("translated formula arity differs from runtime binding ABI")
                    if tuple(formal_names) != tuple(name for name, _ in formula.arg_types):
                        raise WholeArtifactError("translated formula argument names differ from traced ABI")
                    emitter = ModelxCythonRuntimeFormulaEmitter(
                        compiler, typed, fid, refs.sites, formula,
                        object_globals=engine_object_global_indices.get(fid),
                        reference_structural_kinds=reference_structural_kinds(refs),
                    )
                    emitter.emit_function(list(formula.arg_types), analysis_by_id[fid].observed_return_type.dtype)
            except Exception as exc:
                native[fid] = False
                engine_for_family.pop(fid, None)
                engine_rejections[fid] = f"emit:{type(exc).__name__}:{exc}"
        if frozen_candidates:
            from .frozen_references import FrozenRuntimeFormulaEmitter
            for fid in frozen_candidates:
                if not native.get(fid, False):
                    continue
                try:
                    for role, ffid in enumerate(backend.kernel.formula_op_ids):
                        if ffid != fid:
                            continue
                        emitter = FrozenRuntimeFormulaEmitter(
                            compiler, typed, fid, refs.sites,
                            frozen_sites=frozen_plan.sites_for_formula(fid),
                            reference_structural_kinds=reference_structural_kinds(refs),
                        )
                        formal_names = [a.arg for a in emitter.fn.args.args]
                        rows = sorted(sites_by_role.get(role, ()), key=lambda x: x.arg_index)
                        if len(formal_names) != len(rows):
                            raise WholeArtifactError("frozen FormulaOp arity differs from runtime binding ABI")
                        emitter.emit_function(
                            [(name, site.cython_type) for name, site in zip(formal_names, rows)],
                            analysis_by_id[fid].observed_return_type.dtype,
                        )
                    frozen_native[fid] = bool(native.get(fid, False))
                except Exception:
                    native[fid] = False
                    frozen_native[fid] = False
        family_backend[family.family_id] = backend
        family_sites[family.family_id] = sites
        family_native[family.family_id] = native
        family_frozen[family.family_id] = frozen_native
        family_engine[family.family_id] = engine_for_family
        family_static_objects[family.family_id] = static_objects_for_family
        family_references[family.family_id] = refs
        for ordinal, inst in enumerate(backend.instances):
            family_ordinal_by_block[(family.family_id, inst.block_index)] = ordinal

    integrated_specs: dict[int, Any] = {}
    integrated_failures: dict[int, str] = {}
    if experimental_register_regions:
        from .optimized_execution import RegisterRegionPlan
        from .register_integration import prepare_integrated_register_region
        optimized = compiler.optimized_plan or compiler.plan_optimized_execution()
        selected_register_blocks = (
            None if experimental_register_region_blocks is None
            else {int(x) for x in experimental_register_region_blocks}
        )
        for optimized_region in optimized.regions:
            if not isinstance(optimized_region, RegisterRegionPlan):
                continue
            if selected_register_blocks is not None:
                if optimized_region.block_index not in selected_register_blocks:
                    continue
            else:
                from .register_integration import register_region_auto_cost_gate
                accepted, reason = register_region_auto_cost_gate(optimized_region)
                if not accepted:
                    integrated_failures[optimized_region.block_index] = str(reason)
                    continue
            try:
                spec = prepare_integrated_register_region(
                    compiler, optimized_region.block_index, typed, callbacks, analysis
                )
            except Exception as exc:
                integrated_failures[optimized_region.block_index] = f"{type(exc).__name__}: {exc}"
                continue
            integrated_specs[optimized_region.block_index] = spec

    fully_integrated_families = {
        family_id for family_id, backend in family_backend.items()
        if backend.instances and all(inst.block_index in integrated_specs for inst in backend.instances)
    }

    region_data: list[Any] = []
    binding_payloads: list[Any] = []
    literal_specs_by_block: dict[int, list[_LiteralEmitSpec]] = {}
    literal_function_lines: list[str] = []
    family_ops = literal_ops = 0
    literal_native_ops = 0
    byid = compiler.sequential.node_by_id
    role_to_fid = {op.role: op.op_id for op in plan.formula_ops}

    for block_index, block in enumerate(compiler.structured.blocks):
        instance_index = plan.block_to_instance[block_index]
        if isinstance(block, LiteralBlock) or instance_index is None:
            cbs, args, pools, offsets, node_ids, modes, prepared = [], [], [], [], [], [], []
            block_specs: list[_LiteralEmitSpec] = []
            for literal_index, op in enumerate(block.ops):
                node = byid[op.node_id]
                addr = typed.node_by_id.get(node.node_id)
                if addr is None:
                    raise WholeArtifactError(f"literal node {node.node_id} has no physical slot")
                fid = role_to_fid.get(node.shape_token)
                kind = "callback"
                function_name = None
                call_args: tuple[str, ...] = ()
                if (
                    experimental_literal_lowering
                    and fid is not None
                    and fid in frozen_static_object_values
                ):
                    # Static/shared object providers were already proven immutable by
                    # _frozen_static_object_formula_values(). LiteralBlocks used to
                    # ignore that proof and call the original provider every run.
                    kind = "prepared"
                    cbs.append(None)
                    args.append(())
                    modes.append(1)
                    prepared.append(callbacks.wrap_reference(frozen_static_object_values[fid]))
                    literal_native_ops += 1
                else:
                    formula = engine_formula_by_fid.get(fid) if (experimental_literal_lowering and fid is not None) else None
                    if formula is not None and modelx_cython_formula_native_ok(formula, addr.pool):
                        try:
                            if len(formula.arg_types) != len(node.args):
                                raise WholeArtifactError("literal translated formula arity differs from concrete node")
                            call_args = tuple(
                                _literal_call_arg(value, typ)
                                for value, (_name, typ) in zip(node.args, formula.arg_types)
                            )
                            emitter = LiteralModelxCythonFormulaEmitter(
                                compiler, typed, fid, node, formula,
                                object_globals=engine_object_global_indices.get(fid),
                                reference_pool_hints={
                                    name: pool
                                    for (caller_fid, name), pool in engine_reference_pool_hints.items()
                                    if caller_fid == fid
                                },
                            )
                            function_name = f"lit_f_{node.node_id}"
                            literal_function_lines.extend(
                                emitter.emit_literal_function(function_name, list(formula.arg_types))
                            )
                            literal_function_lines.append("")
                            kind = "native"
                            cbs.append(None)
                            args.append(())
                            modes.append(2)
                            prepared.append(None)
                            literal_native_ops += 1
                        except Exception as exc:
                            engine_rejections.setdefault(fid, f"literal_emit:{type(exc).__name__}:{exc}")
                            kind = "callback"
                    if kind == "callback":
                        cbs.append(callbacks.callback_for_obj(node.obj))
                        args.append(tuple(node.args))
                        modes.append(0)
                        prepared.append(None)
                pools.append(_POOL_CODE[addr.pool])
                offsets.append(addr.offset)
                node_ids.append(node.node_id)
                block_specs.append(_LiteralEmitSpec(
                    literal_index, kind, addr.pool, addr.offset, node.node_id,
                    function_name=function_name, call_args=call_args,
                ))
            literal_specs_by_block[block_index] = block_specs
            region_data.append((
                tuple(cbs), tuple(args),
                np.asarray(pools, dtype=np.int64),
                np.asarray(offsets, dtype=np.int64),
                np.asarray(node_ids, dtype=np.int64),
                np.asarray(modes, dtype=np.int64),
                tuple(prepared),
            ))
            literal_ops += len(block.ops)
            continue

        integrated = integrated_specs.get(block_index)
        if integrated is not None:
            region_data.append(integrated.runtime_data)
            binding_payloads.append(integrated.binding_payload)
            family_ops += integrated.operation_count
            continue

        inst = plan.loop_instances[instance_index]
        backend = family_backend[inst.family_id]
        ordinal = family_ordinal_by_block[(inst.family_id, block_index)]
        expansion = expand_native_family_instance(compiler, backend, ordinal)
        sites = family_sites[inst.family_id]
        bind = encode_runtime_bindings(backend.instances[ordinal], sites)
        binding_payloads.append(bind)
        ref = encode_reference_instance(family_references[inst.family_id], ordinal)
        out = _instance_output_descriptors(typed, backend, expansion)
        flat_callbacks = []
        flat_node_ids = []
        role_base = [0]
        for role, node_stream in enumerate(expansion.role_node_ids):
            flat_callbacks.extend(callbacks.callback_for_obj(byid[nid].obj) for nid in node_stream)
            flat_node_ids.extend(int(nid) for nid in node_stream)
            role_base.append(len(flat_node_ids))
        variant_offsets, variant_roles = _runtime_variant_program(backend)
        region_data.append((
            np.asarray(backend.instances[ordinal].variant_ids, dtype=np.int64),
            *bind.as_call_args(),
            *ref.as_call_args(),
            *out,
            tuple(flat_callbacks),
            np.asarray(flat_node_ids, dtype=np.int64),
            np.asarray(role_base, dtype=np.int64),
            variant_offsets,
            variant_roles,
            np.zeros(backend.kernel.role_count, dtype=np.int64),
        ))
        family_ops += expansion.operation_count

    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    module_name = f"{module_prefix}_{abs(hash((id(compiler), compiler.sequential.operation_count))) & 0xffffffff:x}"
    pyx = build_dir / f"{module_name}.pyx"

    lines = [
        "# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True",
        "import numpy as np",
        "cimport numpy as cnp",
        "import math",
        "from libc.math cimport pow as c_pow",
        "",
    ]
    lines += _emit_runtime_helpers()
    lines += [
        "cdef inline Py_ssize_t _missing_literal_reference():",
        "    raise IndexError('literal formula reached a dependency outside the realized graph')",
        "",
    ]
    lines += _emit_runtime_reference_helper()
    if engine_object_global_values:
        lines += _emit_engine_global_support()
    if frozen_plan is not None and frozen_plan.site_count:
        from .frozen_references import emit_frozen_reference_support
        lines += emit_frozen_reference_support(frozen_plan)
    lines += _emit_literal_runner()
    if experimental_literal_lowering and literal_function_lines:
        lines += literal_function_lines
    if experimental_literal_lowering:
        for block_index in sorted(literal_specs_by_block):
            lines += _emit_literal_block_runner(block_index, literal_specs_by_block[block_index])
    if integrated_specs:
        from .register_integration import emit_integrated_register_context_support
        lines += emit_integrated_register_context_support()
    lines += _emit_family_context_type()
    for family_id in sorted(family_backend):
        if family_id in fully_integrated_families:
            continue
        backend = family_backend[family_id]
        sites = family_sites[family_id]
        native = family_native[family_id]
        refs = family_references[family_id]
        ref_structural_kinds = reference_structural_kinds(refs)
        formula_ids = backend.kernel.formula_op_ids
        for fid in dict.fromkeys(formula_ids):
            if not native.get(fid, False):
                continue
            role = formula_ids.index(fid)
            site_rows = sorted((x for x in sites if x.role == role), key=lambda x: x.arg_index)
            static_object_index = family_static_objects.get(family_id, {}).get(fid)
            engine_formula = family_engine.get(family_id, {}).get(fid)
            if static_object_index is not None:
                arg_types = [
                    (f"_arg_{index}", site.cython_type)
                    for index, site in enumerate(site_rows)
                ]
                lines += _emit_frozen_static_object_function(
                    f"f_{family_id}_{fid}", arg_types, static_object_index
                ) + [""]
                continue
            if engine_formula is not None:
                emitter = ModelxCythonRuntimeFormulaEmitter(
                    compiler, typed, fid, refs.sites, engine_formula,
                    object_globals=engine_object_global_indices.get(fid),
                    reference_structural_kinds=ref_structural_kinds,
                )
                arg_types = list(engine_formula.arg_types)
            elif family_frozen.get(family_id, {}).get(fid, False):
                from .frozen_references import FrozenRuntimeFormulaEmitter
                emitter = FrozenRuntimeFormulaEmitter(
                    compiler, typed, fid, refs.sites,
                    frozen_sites=frozen_plan.sites_for_formula(fid),
                    reference_structural_kinds=ref_structural_kinds,
                )
                formal_names = [a.arg for a in emitter.fn.args.args]
                arg_types = [(name, site.cython_type) for name, site in zip(formal_names, site_rows)]
            else:
                emitter = _RuntimeFormulaEmitter(
                    compiler, typed, fid, refs.sites,
                    reference_structural_kinds=ref_structural_kinds,
                )
                formal_names = [a.arg for a in emitter.fn.args.args]
                arg_types = [(name, site.cython_type) for name, site in zip(formal_names, site_rows)]
            emitted = emitter.emit_function(arg_types, analysis_by_id[fid].observed_return_type.dtype)
            emitted = [line.replace(f"f_{fid}(", f"f_{family_id}_{fid}(") for line in emitted]
            lines += emitted + [""]
        if _use_direct_role_dispatch(backend):
            lines += _emit_family_direct_role_functions(
                family_id, backend, sites, typed, compiler, native
            )
        else:
            lines += _emit_family_role_helpers(
                family_id, backend, sites, typed, compiler, native
            )
        lines += _emit_family_runner(
            family_id, backend, sites, typed, compiler, native
        )
    for block_index in sorted(integrated_specs):
        lines += list(integrated_specs[block_index].source_lines)
    lines += [
        "cpdef run_one(cnp.ndarray[cnp.float64_t, ndim=1] d_arr, cnp.ndarray[cnp.int64_t, ndim=1] i_arr, cnp.ndarray[cnp.uint8_t, ndim=1] b_arr, object o_slots, cnp.ndarray[cnp.int64_t, ndim=1] do_arr, cnp.ndarray[cnp.int64_t, ndim=1] io_arr, cnp.ndarray[cnp.int64_t, ndim=1] bo_arr, cnp.ndarray[cnp.int64_t, ndim=1] oo_arr, object regions, bint write_owners):",
        "    cdef double[::1] dslots = d_arr",
        "    cdef long long[::1] islots = i_arr",
        "    cdef unsigned char[::1] bslots = b_arr",
        "    cdef long long[::1] downers = do_arr",
        "    cdef long long[::1] iowners = io_arr",
        "    cdef long long[::1] bowners = bo_arr",
        "    cdef long long[::1] oowners = oo_arr",
    ]
    for block_index in sorted(integrated_specs):
        lines.append(f"    cdef _IntegratedRRContext rr_ctx_{block_index} = <_IntegratedRRContext>regions[{block_index}]")
    for block_index, block in enumerate(compiler.structured.blocks):
        instance_index = plan.block_to_instance[block_index]
        if isinstance(block, LiteralBlock) or instance_index is None:
            if experimental_literal_lowering:
                lines.append(f"    _run_literal_block_{block_index}(regions[{block_index}], dslots, islots, bslots, o_slots, downers, iowners, bowners, oowners, write_owners)")
            else:
                lines.append(f"    _run_literal(regions[{block_index}], dslots, islots, bslots, o_slots, downers, iowners, bowners, oowners, write_owners)")
        elif block_index in integrated_specs:
            spec = integrated_specs[block_index]
            lines.append(f"    {spec.function_name}(rr_ctx_{block_index}, dslots, islots, bslots, o_slots, downers, iowners, bowners, oowners)")
        else:
            inst = plan.loop_instances[instance_index]
            lines.append(f"    _family_{inst.family_id}(regions[{block_index}], dslots, islots, bslots, o_slots, downers, iowners, bowners, oowners, write_owners)")
    lines.append("    return None")
    lines += [
        "",
        "cpdef run_batch(object point_payloads, bint write_owners=True):",
        "    cdef Py_ssize_t i, n = len(point_payloads)",
        "    cdef object p",
        "    for i in range(n):",
        "        p = point_payloads[i]",
        "        run_one(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], write_owners)",
        "    return None",
    ]
    source = "\n".join(lines) + "\n"
    if source_uses_modelx_caches(source):
        raise WholeArtifactError(
            "generated graph artifact references modelx-cython _v_/_has_ cache state"
        )
    pyx.write_text(source)
    mod, so_path, _proc = build_extension(
        pyx, module_name, build_dir, openmp=False, optimization=optimization, native_arch=False
    )
    if engine_object_global_values:
        mod._init_engine_globals(tuple(callbacks.wrap_reference(v) for v in engine_object_global_values))
    if frozen_plan is not None and frozen_plan.site_count:
        mod._init_frozen_references(frozen_plan.payload())
    if integrated_specs:
        for block_index in integrated_specs:
            region_data[block_index] = mod._make_rr_context(region_data[block_index])

    native_formula_pairs = {
        (family_id, fid)
        for family_id, rows in family_native.items()
        for fid, ok in rows.items() if ok
    }
    python_formula_pairs = {
        (family_id, fid)
        for family_id, rows in family_native.items()
        for fid, ok in rows.items() if not ok
    }
    native_operations = literal_native_ops + sum(spec.register_operation_count for spec in integrated_specs.values())
    for block_index, block in enumerate(compiler.structured.blocks):
        instance_index = plan.block_to_instance[block_index]
        if isinstance(block, LiteralBlock) or instance_index is None or block_index in integrated_specs:
            continue
        inst = plan.loop_instances[instance_index]
        backend = family_backend[inst.family_id]
        ordinal = family_ordinal_by_block[(inst.family_id, block_index)]
        expansion = expand_native_family_instance(compiler, backend, ordinal)
        formula_ids = backend.kernel.formula_op_ids
        native_operations += sum(
            1 for role, _nid in expansion.operation_sequence
            if family_native[inst.family_id].get(formula_ids[role], False)
        )
    python_operations = compiler.sequential.operation_count - native_operations

    frozen_operations = 0
    used_frozen_fids: set[int] = set()
    for block_index, block in enumerate(compiler.structured.blocks):
        instance_index = plan.block_to_instance[block_index]
        if isinstance(block, LiteralBlock) or instance_index is None:
            continue
        inst = plan.loop_instances[instance_index]
        backend = family_backend[inst.family_id]
        ordinal = family_ordinal_by_block[(inst.family_id, block_index)]
        expansion = expand_native_family_instance(compiler, backend, ordinal)
        formula_ids = backend.kernel.formula_op_ids
        for role, _nid in expansion.operation_sequence:
            fid = formula_ids[role]
            if family_frozen.get(inst.family_id, {}).get(fid, False):
                frozen_operations += 1
                used_frozen_fids.add(fid)
    used_frozen_sites = [] if frozen_plan is None else [
        site for site in frozen_plan.sites if site.formula_op_id in used_frozen_fids
    ]
    frozen_data_bytes = sum(
        site.values.nbytes + site.axis0.positions.nbytes
        + (0 if site.axis1 is None else site.axis1.positions.nbytes)
        for site in used_frozen_sites
    )

    engine_native_pairs = {
        (family_id, fid)
        for family_id, rows in family_engine.items()
        for fid in rows
        if family_native.get(family_id, {}).get(fid, False)
    }
    engine_native_operations = 0
    for block_index, block in enumerate(compiler.structured.blocks):
        instance_index = plan.block_to_instance[block_index]
        if isinstance(block, LiteralBlock) or instance_index is None or block_index in integrated_specs:
            continue
        inst = plan.loop_instances[instance_index]
        backend = family_backend[inst.family_id]
        ordinal = family_ordinal_by_block[(inst.family_id, block_index)]
        expansion = expand_native_family_instance(compiler, backend, ordinal)
        formula_ids = backend.kernel.formula_op_ids
        engine_native_operations += sum(
            1 for role, _nid in expansion.operation_sequence
            if (inst.family_id, formula_ids[role]) in engine_native_pairs
        )

    report = WholeArtifactBuildReport(
        operation_count=compiler.sequential.operation_count,
        block_count=len(compiler.structured.blocks),
        family_count=len(family_backend),
        family_instance_count=sum(len(x.instances) for x in family_backend.values()),
        literal_block_count=sum(isinstance(b, LiteralBlock) for b in compiler.structured.blocks),
        source_lines=source.count("\n"),
        source_bytes=len(source.encode("utf-8")),
        native_formula_ops=len(native_formula_pairs),
        python_formula_ops=len(python_formula_pairs),
        native_operations=native_operations,
        python_operations=python_operations,
        pyx_path=str(pyx),
        so_path=str(so_path),
        integrated_register_region_count=len(integrated_specs),
        integrated_register_operations=sum(x.register_operation_count for x in integrated_specs.values()),
        integrated_register_python_boundaries=sum(x.python_boundary_count for x in integrated_specs.values()),
        integrated_register_fallback_count=len(integrated_failures),
        frozen_reference_site_count=len(used_frozen_sites),
        frozen_formula_op_count=len(used_frozen_fids),
        frozen_boundary_occurrences=frozen_operations,
        frozen_data_bytes=frozen_data_bytes,
        frozen_prepare_seconds=(0.0 if frozen_plan is None else float(getattr(compiler, "frozen_reference_prepare_seconds", 0.0))),
        frozen_rejected_site_count=(0 if frozen_plan is None else len(frozen_plan.rejected)),
        execution_profile=str(execution_profile),
        target_value_kinds=tuple(_value_kind(v) for v in compiler.trace.target_values),
        storage_baseline_shallow_bytes=int(compiler.storage.plan.baseline_shallow_value_bytes),
        storage_peak_shallow_bytes=int(compiler.storage.plan.peak_live_shallow_value_bytes),
        storage_shallow_value_reduction=float(compiler.storage.plan.exact_shallow_value_reduction),
        modelx_cython_formula_catalog_size=(
            0 if modelx_cython_formula_catalog is None else len(modelx_cython_formula_catalog.formulas)
        ),
        modelx_cython_formula_matched=len(engine_catalog_matched_fids),
        modelx_cython_formula_native=len({fid for _family, fid in engine_native_pairs}),
        modelx_cython_native_operations=engine_native_operations,
        modelx_cython_formula_rejections=tuple(sorted(engine_rejections.items())),
        frozen_static_object_formula_ops=len(frozen_static_object_indices),
        frozen_static_object_operations=sum(
            1
            for node in compiler.trace.nodes
            if node.shape_token in {
                plan.formula_ops[fid].role for fid in frozen_static_object_indices
            }
        ),
        direct_role_dispatch_family_count=sum(
            _use_direct_role_dispatch(backend) for backend in family_backend.values()
        ),
        direct_role_dispatch_role_count=sum(
            backend.kernel.role_count for backend in family_backend.values()
            if _use_direct_role_dispatch(backend)
        ),
        reference_structural_specialized_site_count=sum(
            sum(kind >= 0 for kind in reference_structural_kinds(refs))
            for refs in family_references.values()
        ),
        reference_structural_dynamic_site_count=sum(
            sum(kind < 0 for kind in reference_structural_kinds(refs))
            for refs in family_references.values()
        ),
        formula_catalog_source=modelx_cython_formula_catalog.source_kind,
    )
    return WholeArtifactProgram(
        compiler=compiler,
        typed=typed,
        module=mod,
        region_data=tuple(region_data),
        callback_factory=callbacks,
        binding_payloads=tuple(binding_payloads),
        dslots=dslots,
        islots=islots,
        bslots=bslots,
        oslots=oslots,
        downers=downers,
        iowners=iowners,
        bowners=bowners,
        oowners=oowners,
        targets=targets,
        expected_targets=expected_targets,
        build_report=report,
        family_operations=family_ops,
        literal_operations=literal_ops,
        register_region_failures=tuple(sorted(integrated_failures.items())),
    )
