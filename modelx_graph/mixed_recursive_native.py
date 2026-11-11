from __future__ import annotations

"""Architecture-B mixed scalar/ranked recursive ABI.

Native callers always invoke ``r_<cell>()``.  For N0/N1 Cells the wrapper's
cache miss calls a native ``calc_<cell>()`` formula.  For a bridgeable P Cell
the same native memo wrapper calls an internal Python-backed ``calc_<cell>()``
only on a native-cache miss. Ranked-array and opaque-object P results use the
same native-key cache policy with compiler-owned ``PyObject*`` value slots.
The Python distinction is deliberately hidden behind the recursive ABI.
"""

import ast
import builtins
import copy
import numbers
import operator
import textwrap
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from .dense_recursive_native import (
    DenseMemoCell,
    DenseRecursiveCythonGenerator,
    DenseRecursiveProgram,
    build_dense_recursive_program,
)
from .execution_classes import (
    N0,
    N1,
    P,
    NATIVE_SCALAR_VALUE_ABI,
    PYTHON_OBJECT_VALUE_ABI,
    RANKED_ARRAY_VALUE_ABI,
    ExecutionClassPlan,
    build_execution_class_plan,
)
from .native_formula import NativeFormulaCell, NativeFormulaProgram
from .object_memo_native import ObjectMemoCythonEmitter, ObjectMemoPlan, build_object_memo_plan
from .python_key_memo_native import (
    PythonKeyMemoCythonEmitter,
    PythonKeyMemoPlan,
    build_python_key_memo_plan,
)
from .cython_expression import infer_local_types
from .source_preserving_native import (
    SourcePreservingCompilePlan,
    SourcePreservingDenseCythonGenerator,
    _ModelxLeafRewriter,
    SourcePreservingNativeError,
    build_source_preserving_compile_plan,
    derive_complete_input_numeric_bounds,
)


class MixedRecursiveNativeError(RuntimeError):
    pass


class FallbackTargetResolutionError(MixedRecursiveNativeError):
    """Fail-closed error for an unresolved Python-backed Cell target."""

    def __init__(self, code: str, uid: str, detail: str):
        self.code = str(code)
        self.uid = str(uid)
        self.detail = str(detail)
        super().__init__(f"{self.code}:{self.uid}:{self.detail}")


_NATIVE_DTYPES = {"bool", "int64", "float64"}
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


_DEV136_SAFE_PYTHON_CALLS = {
    "abs", "all", "any", "bool", "float", "int", "len", "max", "min",
    "range", "round", "sum",
}


def _category_helper_name(kind: str, domain_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(domain_id))
    return f"__mx_cat_{kind}_{safe}"


class _CoreCacheCategoryBoundaryRewriter(ast.NodeTransformer):
    """Make only native category representation boundaries explicit.

    The Python formula keeps its original labels/conditionals.  A call into the
    core recursive ABI encodes categorical arguments to int64, while a
    categorical native result is decoded back to its Python label.  Cython owns
    every other expression/control-flow construct.
    """

    def __init__(self, catalog: Mapping[str, NativeFormulaCell]):
        self.catalog = catalog
        self.used_domains: set[str] = set()

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id not in self.catalog:
            return node
        target = self.catalog[node.func.id]
        if node.keywords:
            raise MixedRecursiveNativeError(
                f"{target.source_fullname}: core-cache Cython P call retained keywords"
            )
        if len(node.args) != len(target.parameters):
            raise MixedRecursiveNativeError(
                f"{target.source_fullname}: core-cache Cython P call arity mismatch"
            )
        args: list[ast.expr] = []
        domains = tuple(target.parameter_category_domains)
        if len(domains) < len(node.args):
            domains = domains + (None,) * (len(node.args) - len(domains))
        for arg, did in zip(node.args, domains):
            if did is None:
                args.append(arg)
                continue
            self.used_domains.add(did)
            args.append(ast.Call(ast.Name(_category_helper_name("encode", did), ast.Load()), [arg], []))
        call = ast.copy_location(ast.Call(copy.deepcopy(node.func), args, []), node)
        did = target.result_category_domain
        if did is None:
            return call
        self.used_domains.add(did)
        return ast.copy_location(
            ast.Call(ast.Name(_category_helper_name("decode", did), ast.Load()), [call], []),
            node,
        )


@dataclass(frozen=True)
class PythonReferenceSpec:
    """One non-Cell Python reference needed by a detached P formula.

    Provider calls whose arguments are literals or formula parameters with
    proven singleton ranges are materialized once per run-key during module
    binding, outside recursive execution.  The detached formula receives the
    resulting Python object directly; dynamic provider arguments remain
    fail-closed.
    """

    variable_name: str
    owner_uid: str
    root_name: str
    attribute_path: tuple[str, ...] = ()
    invoke_zero_arg: bool = False
    invoke_args: tuple[Any, ...] = ()
    invoke_kwargs: tuple[tuple[str, Any], ...] = ()

    @property
    def invokes_provider(self) -> bool:
        return bool(self.invoke_zero_arg or self.invoke_args or self.invoke_kwargs)

    @property
    def key(self) -> str:
        suffix = ".".join((self.root_name, *self.attribute_path))
        if self.invoke_zero_arg:
            mode = "call0"
        elif self.invoke_args or self.invoke_kwargs:
            mode = f"call:{self.invoke_args!r}:{self.invoke_kwargs!r}"
        else:
            mode = "ref"
        return f"{self.owner_uid}:{mode}:{suffix}"

    def manifest(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "variable_name": self.variable_name,
            "owner_uid": self.owner_uid,
            "root_name": self.root_name,
            "attribute_path": list(self.attribute_path),
            "invoke_zero_arg": self.invoke_zero_arg,
            "invoke_args": list(self.invoke_args),
            "invoke_kwargs": [[name, value] for name, value in self.invoke_kwargs],
            "invokes_provider": self.invokes_provider,
        }


def _assigned_names(fn: ast.FunctionDef) -> set[str]:
    names = {arg.arg for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)}
    if fn.args.vararg is not None:
        names.add(fn.args.vararg.arg)
    if fn.args.kwarg is not None:
        names.add(fn.args.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                # ``import a.b`` binds ``a`` unless an explicit alias is used.
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not fn:
            names.add(node.name)
    return names


def _attribute_chain(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    attrs: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    return current.id, tuple(reversed(attrs))


class _DetachedPythonReferenceRewriter(ast.NodeTransformer):
    """Replace non-Cell modelx/Python refs with policy-indexed bound objects.

    Cell calls have already been canonicalized to compiler UIDs by the frontend
    and are deliberately left untouched; ``_ModelxLeafRewriter`` later routes
    those calls through ``r_<uid>()``.  This transformer handles only the
    remaining Python object environment.

    A zero-argument call rooted at an external reference is materialized during
    module binding.  This is important for object-valued provider Cells such as
    ``data.return_scenario()``: recursive execution sees the DataFrame itself,
    not a live modelx Cell accessor or a second memoization graph.
    """

    def __init__(
        self,
        *,
        owner_uid: str,
        local_names: set[str],
        catalog: Mapping[str, NativeFormulaCell],
        singleton_parameters: Mapping[str, Any] | None = None,
        reference_aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.owner_uid = owner_uid
        self.local_names = set(local_names)
        self.catalog = catalog
        self.singleton_parameters = dict(singleton_parameters or {})
        self.reference_aliases = dict(reference_aliases or {})
        self._builtin_names = set(dir(builtins))
        self._spec_by_identity: dict[
            tuple[str, tuple[str, ...], bool, tuple[Any, ...], tuple[tuple[str, Any], ...]],
            PythonReferenceSpec,
        ] = {}

    @property
    def specs(self) -> tuple[PythonReferenceSpec, ...]:
        return tuple(sorted(self._spec_by_identity.values(), key=lambda x: x.variable_name))

    def _is_external_root(self, name: str) -> bool:
        return (
            name not in self.local_names
            and name not in self.catalog
            and name not in self._builtin_names
            and not name.startswith("__")
        )

    def _static_provider_arg(self, node: ast.expr) -> Any:
        class _SingletonSubstitution(ast.NodeTransformer):
            def __init__(self, values: Mapping[str, Any]):
                self.values = values

            def visit_Name(self, inner: ast.Name):  # noqa: N802
                if isinstance(inner.ctx, ast.Load) and inner.id in self.values:
                    return ast.copy_location(ast.Constant(self.values[inner.id]), inner)
                return inner

        expr = _SingletonSubstitution(self.singleton_parameters).visit(copy.deepcopy(node))
        ast.fix_missing_locations(expr)
        try:
            value = ast.literal_eval(expr)
        except Exception as exc:
            raise MixedRecursiveNativeError(
                f"{self.owner_uid}: external Python provider argument {ast.unparse(node)!r} "
                "is not a literal or a formula parameter with a proven singleton range"
            ) from exc
        try:
            hash(value)
        except Exception as exc:
            raise MixedRecursiveNativeError(
                f"{self.owner_uid}: external Python provider argument {value!r} is not hashable"
            ) from exc
        return value

    def _static_provider_call_args(
        self, node: ast.Call
    ) -> tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]]:
        args = tuple(self._static_provider_arg(arg) for arg in node.args)
        kwargs: list[tuple[str, Any]] = []
        for kw in node.keywords:
            if kw.arg is None:
                raise MixedRecursiveNativeError(
                    f"{self.owner_uid}: external Python provider **kwargs cannot be detached"
                )
            kwargs.append((str(kw.arg), self._static_provider_arg(kw.value)))
        return args, tuple(kwargs)

    def _spec(
        self,
        root: str,
        attrs: tuple[str, ...],
        invoke: bool,
        invoke_args: tuple[Any, ...] = (),
        invoke_kwargs: tuple[tuple[str, Any], ...] = (),
    ) -> PythonReferenceSpec:
        # The frontend canonicalizes every Cell call to a stable compiler UID,
        # including object-valued helpers that intentionally remain outside the
        # recursive scalar catalog.  Detached Python binding must resolve such
        # owner-local helpers by their real modelx Cell name, not by the UID
        # spelling that exists only in compiler IR.
        root = self.reference_aliases.get(root, root)
        identity = (root, attrs, invoke, invoke_args, invoke_kwargs)
        existing = self._spec_by_identity.get(identity)
        if existing is not None:
            return existing
        safe_uid = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in self.owner_uid)
        index = len(self._spec_by_identity)
        spec = PythonReferenceSpec(
            variable_name=f"__mx_pyref_{safe_uid}_{index}",
            owner_uid=self.owner_uid,
            root_name=root,
            attribute_path=attrs,
            invoke_zero_arg=bool(invoke and not invoke_args and not invoke_kwargs),
            invoke_args=invoke_args,
            invoke_kwargs=invoke_kwargs,
        )
        self._spec_by_identity[identity] = spec
        return spec

    def visit_Call(self, node: ast.Call):  # noqa: N802
        # Materialize external zero-argument providers before visiting their
        # receiver Name, otherwise ``data`` would be rebound while the method
        # call itself remained live.
        chain = _attribute_chain(node.func)
        if chain is not None:
            root, attrs = chain
            if self._is_external_root(root):
                try:
                    args, kwargs = self._static_provider_call_args(node)
                except MixedRecursiveNativeError:
                    # Dynamic library/module calls (for example
                    # ``np.asarray(x)`` or ``pd.MultiIndex.from_arrays(x)``)
                    # execute inside the detached Python formula.  Bind the
                    # root object, not the dynamic call result.
                    return self.generic_visit(node)
                spec = self._spec(root, attrs, True, args, kwargs)
                return ast.copy_location(ast.Name(spec.variable_name, ast.Load()), node)
        if isinstance(node.func, ast.Name) and self._is_external_root(node.func.id):
            try:
                args, kwargs = self._static_provider_call_args(node)
            except MixedRecursiveNativeError:
                return self.generic_visit(node)
            spec = self._spec(node.func.id, (), True, args, kwargs)
            return ast.copy_location(ast.Name(spec.variable_name, ast.Load()), node)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):  # noqa: N802
        if isinstance(node.ctx, ast.Load) and self._is_external_root(node.id):
            spec = self._spec(node.id, (), False)
            return ast.copy_location(ast.Name(spec.variable_name, ast.Load()), node)
        return node


class _PolicyIndexedPythonReferenceRewriter(ast.NodeTransformer):
    def __init__(
        self, variable_names: set[str], policy_index: str | ast.expr
    ) -> None:
        self.variable_names = set(variable_names)
        self.policy_index = policy_index

    def _index(self) -> ast.expr:
        if isinstance(self.policy_index, str):
            return ast.Name(self.policy_index, ast.Load())
        return copy.deepcopy(self.policy_index)

    def visit_Name(self, node: ast.Name):  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id in self.variable_names:
            return ast.copy_location(
                ast.Subscript(
                    ast.Name(node.id, ast.Load()),
                    self._index(),
                    ast.Load(),
                ),
                node,
            )
        return node


class _ObjectMemoCallRewriter(ast.NodeTransformer):
    """Route object-valued Cell calls through compiler-owned object memos."""

    def __init__(
        self,
        cells: Mapping[str, NativeFormulaCell],
        policy_index: str | ast.expr,
    ) -> None:
        self.cells = dict(cells)
        self.policy_index = policy_index

    def _index(self) -> ast.expr:
        if isinstance(self.policy_index, str):
            return ast.Name(self.policy_index, ast.Load())
        return copy.deepcopy(self.policy_index)

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id not in self.cells:
            return node
        target = self.cells[node.func.id]
        if node.keywords or len(node.args) != target.arity:
            raise MixedRecursiveNativeError(
                f"{target.source_fullname}: object memo call requires "
                f"{target.arity} positional arguments"
            )
        return ast.copy_location(
            ast.Call(
                ast.Name(f"r_{target.uid}", ast.Load()),
                [self._index(), *node.args],
                [],
            ),
            node,
        )


class _PythonKeyMemoCallRewriter(ast.NodeTransformer):
    """Route detached helper Cell calls through Python-key ``r_<uid>()``.

    This rewriter is used only inside GIL-held detached Python formula bodies.
    Native N0/N1 bodies must never call these wrappers directly.
    """

    def __init__(
        self,
        cells: Mapping[str, NativeFormulaCell],
        policy_index: str | ast.expr,
    ) -> None:
        self.cells = dict(cells)
        self.policy_index = policy_index

    def _index(self) -> ast.expr:
        if isinstance(self.policy_index, str):
            return ast.Name(self.policy_index, ast.Load())
        return copy.deepcopy(self.policy_index)

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id not in self.cells:
            return node
        target = self.cells[node.func.id]
        if node.keywords or len(node.args) != target.arity:
            raise MixedRecursiveNativeError(
                f"{target.source_fullname}: Python-key memo call requires "
                f"{target.arity} positional arguments"
            )
        return ast.copy_location(
            ast.Call(
                ast.Name(f"r_{target.uid}", ast.Load()),
                [self._index(), *node.args],
                [],
            ),
            node,
        )


class _RankedNativeMemoCallRewriter(ast.NodeTransformer):
    """Route ranked/GIL formula dependencies through rank-local wrappers."""

    def __init__(self, cells: Mapping[str, NativeFormulaCell]) -> None:
        self.cells = dict(cells)

    def visit_Call(self, node: ast.Call):  # noqa: N802
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id not in self.cells:
            return node
        target = self.cells[node.func.id]
        if node.keywords or len(node.args) != target.arity:
            raise MixedRecursiveNativeError(
                f"{target.source_fullname}: ranked memo call requires "
                f"{target.arity} positional arguments"
            )
        return ast.copy_location(
            ast.Call(ast.Name(f"r_{target.uid}", ast.Load()), node.args, []),
            node,
        )


class _RestorePythonGuardRewriter(ast.NodeTransformer):
    """Restore original Python exceptions inside detached P formula bodies.

    The frontend canonicalizes explicit ``raise`` statements to
    ``__guard_fail__(code)`` so native N0/N1 code can report failures outside
    ``nogil``. A P formula executes under the GIL and should preserve ordinary
    Python semantics instead. Its existing guard table is sufficient authority
    to reconstruct the original exception type and message.
    """

    def __init__(self, guards: Sequence[Any]) -> None:
        self.guards = {int(g.code): g for g in guards}

    def visit_Expr(self, node: ast.Expr):  # noqa: N802
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "__guard_fail__"
            and len(value.args) == 1
            and not value.keywords
            and isinstance(value.args[0], ast.Constant)
            and isinstance(value.args[0].value, int)
        ):
            guard = self.guards.get(int(value.args[0].value))
            if guard is not None:
                exc = ast.Call(
                    ast.Name(str(guard.exception_type), ast.Load()),
                    [ast.Constant(str(guard.message))],
                    [],
                )
                return ast.copy_location(ast.Raise(exc=exc, cause=None), node)
        return self.generic_visit(node)


def _owner_local_reference_aliases(plan: Any, cell: NativeFormulaCell) -> dict[str, str]:
    """Map canonical IR UIDs for owner-local Cells back to modelx names.

    Scalar recursive Cells are already present in ``catalog`` and therefore do
    not pass through detached-reference binding.  This map is for object-valued
    or otherwise non-recursive helper Cells that the frontend canonicalized to
    ``mc_<digest>_<name>`` but that still need preparation-time Python binding.

    Only Cells on the P formula's own exported modelx class are admitted.  That
    keeps resolution deterministic and fail-closed instead of guessing how to
    navigate cross-space references.
    """
    module_info = getattr(plan, "module_info", None)
    module_name = getattr(plan, "module_name", None)
    classes = getattr(module_info, "classes", None)
    if module_info is None or not module_name or not isinstance(classes, Mapping):
        return {}

    owner_name: str | None = None
    owner_info: Any | None = None
    for class_name, class_info in classes.items():
        fqname = str(getattr(class_info, "fqname", "") or "")
        if fqname and str(cell.source_fullname).startswith(fqname + "."):
            owner_name = str(class_name)
            owner_info = class_info
            break
    if owner_name is None or owner_info is None:
        return {}

    owner_cells = getattr(owner_info, "cells", None)
    if not isinstance(owner_cells, Mapping):
        return {}

    # Import lazily: the fast backend imports this module from plan methods.
    # At candidate-build time the backend module is fully initialized.
    from .modelx_cython_fast_backend import _uid as _frontend_uid

    return {
        _frontend_uid(str(module_name), owner_name, str(source_name)): str(source_name)
        for source_name in owner_cells
    }


def _python_semantics_lossy_ranked_provider(plan: Any, cell: NativeFormulaCell) -> bool:
    """Return whether a ranked leaf's native payload loses Python surface semantics.

    Prepared structured columns may be encoded as int64 for native execution even
    when their source values are categorical/datetime pandas objects.  A detached
    Python formula must not receive that physical array when it expects Python
    behaviour such as ``Series.dt``.  Keep the native view for native callers, but
    let the Python formula bind the original zero-argument provider value.
    """
    fn = cell.function
    if fn is None:
        return False
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    value = body[0].value
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in {"point_input", "array_input", "table_input"}
        and value.args
        and isinstance(value.args[0], ast.Constant)
        and isinstance(value.args[0].value, str)
    ):
        return False
    specs = dict(getattr(plan, "source_preserving_runtime_inputs", {}) or {})
    spec = specs.get(value.args[0].value)
    structured = getattr(spec, "normalized_structured_column", None)
    if structured is None:
        return False
    logical_kind = str(getattr(structured, "logical_kind", ""))
    source_dtype = str(getattr(structured, "source_dtype", ""))
    if logical_kind == "category":
        return True
    numeric_source = (
        source_dtype.startswith("int")
        or source_dtype.startswith("uint")
        or source_dtype.startswith("float")
        or source_dtype in {"bool", "boolean"}
    )
    return not numeric_source



def _native_detached_recursive_catalog(
    plan: Any,
    catalog: Mapping[str, NativeFormulaCell],
) -> dict[str, NativeFormulaCell]:
    """Cells that already own the existing native/object recursive ABI."""
    classes = getattr(plan, "execution_class_plan", None)
    if classes is None:
        return {}
    return {
        dep_uid: dep_cell
        for dep_uid, dep_cell in catalog.items()
        if dep_cell.execution_role == "formula"
        and dep_uid in classes.cells
        and classes.cell(dep_uid).bridgeable_python_fallback
        and classes.cell(dep_uid).value_abi in {
            NATIVE_SCALAR_VALUE_ABI,
            PYTHON_OBJECT_VALUE_ABI,
            RANKED_ARRAY_VALUE_ABI,
        }
        and not _python_semantics_lossy_ranked_provider(plan, dep_cell)
    }


def _has_category_boundary(cell: NativeFormulaCell) -> bool:
    return bool(
        cell.result_category_domain is not None
        or any(did is not None for did in cell.parameter_category_domains)
    )


def _python_key_helper_candidate(
    plan: Any,
    uid: str,
    catalog: Mapping[str, NativeFormulaCell],
    python_key_catalog: Mapping[str, NativeFormulaCell],
) -> tuple[NativeFormulaCell, tuple[str, ...], tuple[PythonReferenceSpec, ...]] | None:
    """Return one detached helper whose *key* requires Python semantics.

    Python-key helpers are P Cells whose call keys require ordinary Python
    semantics. Their formula body remains under the GIL and helper-to-helper
    calls remain compiler-owned Cell calls. Calls into the native recursive ABI
    use the same reversible category boundary as core-cache detached P Cells:
    Python labels are encoded before native calls and categorical native results
    are decoded back into Python values.
    """
    classes = getattr(plan, "execution_class_plan", None)
    raw_functions = dict(getattr(plan, "formula_source_functions", {}))
    if classes is None or uid not in raw_functions or uid not in catalog:
        return None
    row = classes.cell(uid)
    cell = catalog[uid]
    if row.execution_class != P or row.bridgeable_python_fallback:
        return None
    if cell.execution_role != "formula" or cell.logical_result_ndim != 0:
        return None
    fn = copy.deepcopy(raw_functions[uid])
    if not isinstance(fn, ast.FunctionDef):
        return None
    fn = _RestorePythonGuardRewriter(cell.guards).visit(fn)
    assert isinstance(fn, ast.FunctionDef)

    native_catalog = _native_detached_recursive_catalog(plan, catalog)
    callable_catalog = dict(native_catalog)
    callable_catalog.update(python_key_catalog)

    # Python-key helpers execute under the GIL and therefore keep Python
    # category values internally.  Make the representation change explicit only
    # at calls into the native recursive ABI.  The domain metadata comes from
    # the existing proven category plan; no helper-local/product-specific domain
    # inference is performed here.
    bridge = _CoreCacheCategoryBoundaryRewriter(native_catalog)
    try:
        fn = bridge.visit(fn)
    except MixedRecursiveNativeError:
        return None
    assert isinstance(fn, ast.FunctionDef)

    singleton_parameters: dict[str, Any] = {}
    bounds = dict(getattr(plan, "parameter_ranges", {}) or {}).get(uid)
    if bounds is not None and len(bounds) == len(cell.parameters):
        for name, bound in zip(cell.parameters, bounds):
            if bound is None or len(bound) != 2:
                continue
            lo, hi = bound
            if lo == hi:
                singleton_parameters[name] = lo

    owner_local_aliases = _owner_local_reference_aliases(plan, cell)
    refs = _DetachedPythonReferenceRewriter(
        owner_uid=uid,
        local_names=_assigned_names(fn),
        catalog=callable_catalog,
        singleton_parameters=singleton_parameters,
        reference_aliases=owner_local_aliases,
    )
    try:
        fn = refs.visit(fn)
    except MixedRecursiveNativeError:
        return None
    assert isinstance(fn, ast.FunctionDef)
    ast.fix_missing_locations(fn)

    owner_local_cell_names = set(owner_local_aliases.values())
    if any(
        spec.root_name in owner_local_cell_names and not spec.invokes_provider
        for spec in refs.specs
    ):
        return None

    deps = {
        node.func.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in callable_catalog
    }
    for dep in deps:
        if dep in python_key_catalog:
            continue
        dep_row = classes.cell(dep)
        if not dep_row.bridgeable_python_fallback:
            return None

    return (
        replace(cell, function=fn, reduction=None, ranked_function=None),
        tuple(sorted(bridge.used_domains)),
        refs.specs,
    )


def _python_key_helper_candidates(
    plan: Any,
    catalog: Mapping[str, NativeFormulaCell],
) -> tuple[
    Mapping[str, NativeFormulaCell],
    Mapping[str, tuple[str, ...]],
    Mapping[str, tuple[PythonReferenceSpec, ...]],
]:
    """Find the fixed point of non-native-key detached helper Cells.

    Candidate discovery is intentionally independent from selected-closure
    traversal.  A helper may call another helper recursively, so the provisional
    catalog is pruned until every retained body can be detached without a live
    modelx callable.
    """
    classes = getattr(plan, "execution_class_plan", None)
    raw_functions = dict(getattr(plan, "formula_source_functions", {}))
    if classes is None:
        return {}, {}, {}

    current: dict[str, NativeFormulaCell] = {
        uid: catalog[uid]
        for uid, row in classes.cells.items()
        if row.execution_class == P
        and not row.bridgeable_python_fallback
        and uid in catalog
        and uid in raw_functions
        and catalog[uid].execution_role == "formula"
        and catalog[uid].logical_result_ndim == 0
        and isinstance(raw_functions[uid], ast.FunctionDef)
    }
    domains_by_uid: dict[str, tuple[str, ...]] = {}
    refs_by_uid: dict[str, tuple[PythonReferenceSpec, ...]] = {}
    while current:
        built: dict[str, NativeFormulaCell] = {}
        built_domains: dict[str, tuple[str, ...]] = {}
        built_refs: dict[str, tuple[PythonReferenceSpec, ...]] = {}
        for uid in sorted(current):
            candidate = _python_key_helper_candidate(plan, uid, catalog, current)
            if candidate is None:
                continue
            cell, domains, refs = candidate
            built[uid] = cell
            built_domains[uid] = tuple(domains)
            built_refs[uid] = tuple(refs)
        if set(built) == set(current):
            return built, built_domains, built_refs
        current = {uid: catalog[uid] for uid in built}
        domains_by_uid = built_domains
        refs_by_uid = built_refs
    return {}, domains_by_uid, refs_by_uid


def _core_cache_cython_p_candidate(
    plan: Any,
    uid: str,
    catalog: Mapping[str, NativeFormulaCell],
    *,
    python_key_catalog: Mapping[str, NativeFormulaCell] | None = None,
) -> tuple[NativeFormulaCell, tuple[str, ...], tuple[PythonReferenceSpec, ...]] | None:
    """Return a detached Python-semantic P formula backed by native Cell caches.

    P means only that this formula body needs Python semantics.  It does *not*
    authorize a second modelx execution engine.  The original formula source is
    retained under the GIL, every scalar Cell dependency is routed through the
    unified ``r_<uid>()`` native memo ABI, and any remaining non-Cell Python
    references are bound explicitly at module initialization.

    Object-valued zero-argument providers are materialized before recursive
    execution, so DataFrame/Series operations remain ordinary Python operations
    without invoking modelx Cells from inside the projection loop.
    """
    classes = getattr(plan, "execution_class_plan", None)
    raw_functions = dict(getattr(plan, "formula_source_functions", {}))
    if classes is None or uid not in raw_functions or uid not in catalog:
        return None
    row = classes.cell(uid)
    cell = catalog[uid]
    if row.execution_class != P or not row.bridgeable_python_fallback:
        return None
    if (
        cell.execution_role != "formula"
        or row.value_abi not in {
            NATIVE_SCALAR_VALUE_ABI,
            PYTHON_OBJECT_VALUE_ABI,
            RANKED_ARRAY_VALUE_ABI,
        }
    ):
        return None
    fn = copy.deepcopy(raw_functions[uid])
    if not isinstance(fn, ast.FunctionDef):
        return None
    fn = _RestorePythonGuardRewriter(cell.guards).visit(fn)
    assert isinstance(fn, ast.FunctionDef)

    # Only recursive formula Cells have an r_<uid> implementation.  A
    # materialized helper appearing in preserved Python source is an external
    # object provider and must be bound before execution, not left as a call to
    # a wrapper that the ranked/scalar generators never emit.
    # A Python formula may refer to owner-local helper Cells that deliberately
    # do not have a recursive compiler ABI (for example a DataFrame/Series
    # provider).  modelx-cython keeps such Python execution at the Python level
    # and copies only the references needed by the executing formula.  Mirror
    # that ownership rule here: only Cells with a proven compiler-callable ABI
    # remain recursive calls.  Other owner-local Cells are eligible to become
    # pre-bound provider values when their call arguments are static for the
    # run key; dynamic calls remain fail-closed below.
    recursive_catalog = _native_detached_recursive_catalog(plan, catalog)
    python_key_catalog = dict(python_key_catalog or {})
    callable_catalog = dict(recursive_catalog)
    callable_catalog.update(python_key_catalog)

    bridge = _CoreCacheCategoryBoundaryRewriter(recursive_catalog)
    fn = bridge.visit(fn)
    assert isinstance(fn, ast.FunctionDef)
    # The recursive ABI stores categories as int64 codes, while the detached
    # Python body should observe the original Python labels.  Calls made by the
    # body are handled by the boundary rewriter above; the formula's own
    # arguments/result are decoded/encoded by the emitter.
    bridge.used_domains.update(
        did for did in cell.parameter_category_domains if did is not None
    )
    if cell.result_category_domain is not None:
        bridge.used_domains.add(cell.result_category_domain)

    singleton_parameters: dict[str, Any] = {}
    bounds = dict(getattr(plan, "parameter_ranges", {}) or {}).get(uid)
    if bounds is not None and len(bounds) == len(cell.parameters):
        for name, bound in zip(cell.parameters, bounds):
            if bound is None or len(bound) != 2:
                continue
            lo, hi = bound
            if lo == hi:
                singleton_parameters[name] = lo

    owner_local_aliases = _owner_local_reference_aliases(plan, cell)
    refs = _DetachedPythonReferenceRewriter(
        owner_uid=uid,
        local_names=_assigned_names(fn),
        catalog=callable_catalog,
        singleton_parameters=singleton_parameters,
        reference_aliases=owner_local_aliases,
    )
    try:
        fn = refs.visit(fn)
    except MixedRecursiveNativeError:
        return None
    assert isinstance(fn, ast.FunctionDef)
    ast.fix_missing_locations(fn)

    # Never smuggle an owner-local modelx Cell callable into recursive
    # execution.  If a non-recursive Cell call could be materialized from
    # literal/singleton arguments, the rewriter replaced the whole call with a
    # provider value and ``invokes_provider`` is true.  Otherwise it would have
    # rebound the Cell object itself, which would recreate a live-modelx
    # execution path after the model is closed.  Reject that candidate.
    owner_local_cell_names = set(owner_local_aliases.values())
    if any(
        spec.root_name in owner_local_cell_names and not spec.invokes_provider
        for spec in refs.specs
    ):
        return None

    deps = tuple(sorted({
        node.func.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in callable_catalog
    }))
    for dep in deps:
        if dep in python_key_catalog:
            continue
        dep_row = classes.cell(dep)
        if not dep_row.bridgeable_python_fallback:
            return None

    return (
        replace(cell, function=fn, reduction=None, ranked_function=None),
        tuple(sorted(bridge.used_domains)),
        refs.specs,
    )


@dataclass(frozen=True)
class FallbackSpaceHop:
    """One metadata-proven transition from a caller Space to another Space."""

    receiver_path: tuple[str, ...]
    target_class_fullname: str
    target_space_parameters: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "receiver_path": list(self.receiver_path),
            "target_class_fullname": self.target_class_fullname,
            "target_space_parameters": list(self.target_space_parameters),
        }


@dataclass(frozen=True)
class FallbackTargetSpec:
    """Compiler-owned identity metadata for one original Python-backed Cell."""

    uid: str
    source_name: str
    source_fullname: str
    owner_class_fullname: str
    resolution_kind: str  # same_space | cross_space
    hops: tuple[FallbackSpaceHop, ...] = ()
    identity_authority: str = "modelx_cython.CellsResolver+ModuleInfo+canonical_Cell_source_identity"

    def manifest(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source_name": self.source_name,
            "source_fullname": self.source_fullname,
            "owner_class_fullname": self.owner_class_fullname,
            "resolution_kind": self.resolution_kind,
            "hops": [hop.manifest() for hop in self.hops],
            "identity_authority": self.identity_authority,
        }


@dataclass(frozen=True)
class FallbackTargetBlocker:
    code: str
    uid: str
    source_fullname: str
    detail: str

    def manifest(self) -> dict[str, str]:
        return {
            "code": self.code,
            "uid": self.uid,
            "source_fullname": self.source_fullname,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FallbackTargetPlan:
    """Static resolver metadata for every P Cell in the recursive ABI closure."""

    schema: str
    root_class_fullname: str
    targets: Mapping[str, FallbackTargetSpec]
    blockers: tuple[FallbackTargetBlocker, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.blockers

    def target(self, uid: str) -> FallbackTargetSpec:
        try:
            return self.targets[uid]
        except KeyError as exc:
            raise FallbackTargetResolutionError(
                "P_TARGET_UNRESOLVED", uid, "no compiler-owned fallback target spec"
            ) from exc

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "supported": self.supported,
            "root_class_fullname": self.root_class_fullname,
            "targets": {
                uid: spec.manifest() for uid, spec in sorted(self.targets.items())
            },
            "blockers": [row.manifest() for row in self.blockers],
        }


def _cell_owner_class_fullname(cell: NativeFormulaCell) -> str:
    fullname = str(cell.source_fullname or "")
    if "." not in fullname:
        return ""
    return fullname.rsplit(".", 1)[0]


def _normalized_receiver_path(receiver_chain: Sequence[str]) -> tuple[str, ...]:
    # The final element names the Cell itself.  Everything before it identifies
    # the target Space relative to the caller Space.  Exported modelx-cython
    # formulas normally omit ``self``; strip it defensively if present.
    path = tuple(str(x) for x in tuple(receiver_chain)[:-1])
    if path and path[0] in {"self", "cls"}:
        path = path[1:]
    return path


def build_fallback_target_plan(
    plan: Any, *, p_uids: Sequence[str] | None = None
) -> FallbackTargetPlan:
    """Derive original-Cell runtime targets without product-specific wiring.

    Same-Space targets are direct attributes of the active root ItemSpace.
    Cross-Space targets reuse the exact class/call identity already resolved by
    modelx-cython's ``CellsResolver``.  The resulting hop chain is metadata, not
    a second recursive calling convention: generated formulas still call only
    ``r_<cell>()``.
    """

    classes = getattr(plan, "execution_class_plan", None)
    if classes is None:
        return FallbackTargetPlan(
            schema="modelx_graph.fallback_target_plan.v1",
            root_class_fullname="",
            targets={},
            blockers=(FallbackTargetBlocker(
                "P_TARGET_UNRESOLVED", "", "", "fast plan has no execution-class plan"
            ),),
        )

    root_module = str(getattr(plan, "module_name", ""))
    root_class_name = str(getattr(plan, "class_name", ""))
    root_class = ".".join(x for x in (root_module, root_class_name) if x)
    catalog: dict[str, NativeFormulaCell] = dict(
        getattr(plan, "source_transparent_cell_catalog", {})
    )
    if p_uids is None:
        # Detached P formulas still need a compiler-owned owner-space identity
        # even though their Cell body is never invoked through live modelx.
        # Use the same detached-P-aware closure as the mixed program itself:
        # ExecutionClassPlan.recursive_abi_uids stops traversal at P, while a
        # detached P body can call additional native/P Cells through r_<uid>().
        catalog = dict(getattr(plan, "source_transparent_cell_catalog", {}))
        (
            recursive_uids, _gil_backed, _unsupported, _core_cache,
            python_key_cells, _bridge_domains, python_reference_specs,
        ) = _mixed_recursive_uids(plan, catalog)
        # Owner-space metadata is needed not only for P Cells themselves but
        # for every detached Python reference. Ranked/native formulas can retain
        # module/provider references that are bound outside execution too.
        p_uids = tuple(sorted({
            *(uid for uid in recursive_uids if classes.cell(uid).execution_class == P),
            *python_key_cells,
            *(spec.owner_uid for spec in python_reference_specs),
        }))
    else:
        p_uids = tuple(sorted(set(str(uid) for uid in p_uids)))

    # Each path is a sequence of class-boundary hops.  Keeping boundaries (rather
    # than flattening attribute names) lets runtime selection bind parameterized
    # target Spaces generically before continuing through a nested cross-Space path.
    paths_by_class: dict[str, set[tuple[FallbackSpaceHop, ...]]] = {
        root_class: {()}
    }
    records = tuple(getattr(plan, "cross_space_calls", ()))
    ordered_records = tuple(sorted(
        records,
        key=lambda r: (
            str(r.caller_class_fullname), tuple(r.receiver_chain),
            str(r.callee_class_fullname), str(r.callee_uid),
        ),
    ))
    max_paths_per_class = 16
    # Simple paths only. Cross-Space class cycles do not create a new runtime
    # identity for a target already on the path and would otherwise grow forever.
    for _ in range(max(1, len(ordered_records) + 1)):
        changed = False
        for record in ordered_records:
            caller = str(record.caller_class_fullname)
            callee = str(record.callee_class_fullname)
            parent_paths = tuple(paths_by_class.get(caller, ()))
            if not parent_paths:
                continue
            receiver_path = _normalized_receiver_path(record.receiver_chain)
            if not receiver_path:
                continue
            hop = FallbackSpaceHop(
                receiver_path=receiver_path,
                target_class_fullname=callee,
                target_space_parameters=tuple(record.target_space_parameters),
            )
            bucket = paths_by_class.setdefault(callee, set())
            before = len(bucket)
            for parent in parent_paths:
                visited = {root_class, *(x.target_class_fullname for x in parent)}
                if callee in visited:
                    continue
                if len(bucket) >= max_paths_per_class:
                    break
                bucket.add(parent + (hop,))
            changed |= len(bucket) != before
        if not changed:
            break

    targets: dict[str, FallbackTargetSpec] = {}
    blockers: list[FallbackTargetBlocker] = []
    for uid in p_uids:
        cell = catalog.get(uid)
        if cell is None:
            blockers.append(FallbackTargetBlocker(
                "P_TARGET_UNRESOLVED", uid, "", "P Cell is absent from source-transparent catalog"
            ))
            continue
        owner = _cell_owner_class_fullname(cell)
        if not owner:
            blockers.append(FallbackTargetBlocker(
                "P_TARGET_UNRESOLVED", uid, cell.source_fullname,
                "cannot derive owning modelx-cython class from canonical source fullname",
            ))
            continue
        if owner == root_class:
            targets[uid] = FallbackTargetSpec(
                uid=uid,
                source_name=cell.source_name,
                source_fullname=cell.source_fullname,
                owner_class_fullname=owner,
                resolution_kind="same_space",
                hops=(),
            )
            continue
        candidate_paths = tuple(sorted(
            paths_by_class.get(owner, ()),
            key=lambda path: tuple(
                (hop.target_class_fullname, hop.receiver_path, hop.target_space_parameters)
                for hop in path
            ),
        ))
        if not candidate_paths:
            blockers.append(FallbackTargetBlocker(
                "P_TARGET_UNRESOLVED", uid, cell.source_fullname,
                f"no metadata-resolved root path to owning class {owner}",
            ))
            continue
        if len(candidate_paths) != 1:
            rendered = [
                "/".join(".".join(h.receiver_path) for h in path)
                for path in candidate_paths[:4]
            ]
            blockers.append(FallbackTargetBlocker(
                "P_TARGET_AMBIGUOUS", uid, cell.source_fullname,
                f"owning class {owner} has {len(candidate_paths)} runtime paths: {rendered}",
            ))
            continue
        targets[uid] = FallbackTargetSpec(
            uid=uid,
            source_name=cell.source_name,
            source_fullname=cell.source_fullname,
            owner_class_fullname=owner,
            resolution_kind="cross_space",
            hops=candidate_paths[0],
        )

    return FallbackTargetPlan(
        schema="modelx_graph.fallback_target_plan.v1",
        root_class_fullname=root_class,
        targets=targets,
        blockers=tuple(blockers),
    )


def _convert_native_result(value: Any, dtype: str) -> bool | int | float:
    """Checked Python -> native-ABI scalar conversion for a P-cell miss."""
    if dtype == "bool":
        if isinstance(value, bool):
            return bool(value)
        # numpy.bool_ and compatible scalar bools expose a strict bool dtype but
        # are not instances of bool.  Avoid truth-testing arbitrary objects.
        if type(value).__module__.startswith("numpy") and type(value).__name__ == "bool_":
            return bool(value)
        raise TypeError(f"Python-backed Cell returned {type(value).__name__}; expected bool")
    if dtype == "int64":
        try:
            result = operator.index(value)
        except TypeError as exc:
            raise TypeError(
                f"Python-backed Cell returned {type(value).__name__}; expected exact integer"
            ) from exc
        result = int(result)
        if result < _INT64_MIN or result > _INT64_MAX:
            raise OverflowError(f"Python-backed Cell integer result {result} is outside int64")
        return result
    if dtype == "float64":
        if isinstance(value, numbers.Real):
            return float(value)
        # numpy real scalar types register inconsistently across versions.
        if type(value).__module__.startswith("numpy"):
            try:
                result = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(
                    f"Python-backed Cell returned {type(value).__name__}; expected real scalar"
                ) from exc
            return result
        raise TypeError(f"Python-backed Cell returned {type(value).__name__}; expected real scalar")
    raise MixedRecursiveNativeError(f"unsupported Python-backed native result dtype {dtype!r}")


class FallbackPolicyRuntime:
    """Binding owner for detached Python-semantic P execution.

    Target-state mixed execution never calls a modelx formula on an ``r_<P>()``
    miss.  P bodies live in the generated Cython module under the GIL and all
    scalar Cell dependencies call the native recursive ABI.  This runtime now
    primarily resolves the remaining non-Cell Python references once, before a
    batch runs.  ``invoke_native`` is retained only for compatibility with old
    callers/tests and is not selected by newly built mixed programs.
    """

    def __init__(
        self,
        *,
        run_keys: Sequence[Any],
        policy_factory: Callable[[Any], Any],
        target_resolver: Callable[[Any, str, str], Callable[..., Any]],
        clear_policy_state: Callable[[], None] | None = None,
        python_reference_resolver: Callable[[
            int, str, str, tuple[str, ...], bool, tuple[Any, ...], tuple[tuple[str, Any], ...]
        ], Any] | None = None,
    ) -> None:
        self.run_keys = tuple(run_keys)
        self._policy_factory = policy_factory
        self._target_resolver = target_resolver
        self._clear_policy_state = clear_policy_state
        self._python_reference_resolver = python_reference_resolver
        self._active_policy_i: int | None = None
        self._active_context: Any = None
        self._targets: dict[str, Callable[..., Any]] = {}
        self._started = False
        self._python_invocations = 0
        self._policy_activations = 0
        self._policy_discards = 0
        self._per_cell: Counter[str] = Counter()
        self._python_reference_resolutions = 0
        self._python_provider_materializations = 0

    @classmethod
    def from_modelx_space(
        cls,
        *,
        run_keys: Sequence[Any],
        model: Any,
        space: Any,
        source_name_by_uid: Mapping[str, str],
        target_resolvers: Mapping[str, Callable[[Any], Callable[..., Any]]] | None = None,
    ) -> "FallbackPolicyRuntime":
        """Create a same-space modelx/modelx-cython fallback owner.

        ``target_resolvers`` is an escape hatch for cross-space P Cells: a UID
        can resolve its original callable from the active itemspace without
        changing the generated recursive ABI.
        """
        source_name_by_uid = dict(source_name_by_uid)
        target_resolvers = dict(target_resolvers or {})

        def policy_factory(key: Any) -> Any:
            # Unparameterized Spaces use the singleton invocation key ``()`` in
            # ranked plans but are not itemizable.  Detached Python-reference
            # binding should resolve against the Space itself in that case.
            if getattr(space, "formula", None) is None:
                return space
            return space[key]

        def target_resolver(context: Any, uid: str, source_name: str) -> Callable[..., Any]:
            custom = target_resolvers.get(uid)
            if custom is not None:
                target = custom(context)
            else:
                target = getattr(context, source_name_by_uid.get(uid, source_name))
            if not callable(target):
                raise TypeError(f"fallback target {uid!r} is not callable")
            return target

        run_keys = tuple(run_keys)

        def python_reference_resolver(
            policy_i: int,
            uid: str,
            root_name: str,
            attribute_path: tuple[str, ...],
            invoke_zero_arg: bool,
            invoke_args: tuple[Any, ...] = (),
            invoke_kwargs: tuple[tuple[str, Any], ...] = (),
        ) -> Any:
            i = int(policy_i)
            if i < 0 or i >= len(run_keys):
                raise IndexError(
                    f"Python-reference policy index {i} outside [0,{len(run_keys)})"
                )
            context = policy_factory(run_keys[i])
            if root_name in {"self", "cls"}:
                value = context
            else:
                try:
                    value = getattr(context, root_name)
                except Exception as exc:
                    raise FallbackTargetResolutionError(
                        "P_PYREF_RUNTIME_UNRESOLVED",
                        uid,
                        f"active ItemSpace has no reference {root_name!r}",
                    ) from exc
            for attr in attribute_path:
                try:
                    value = getattr(value, attr)
                except Exception as exc:
                    raise FallbackTargetResolutionError(
                        "P_PYREF_RUNTIME_UNRESOLVED",
                        uid,
                        f"cannot resolve Python reference attribute {attr!r} in "
                        f"{root_name}.{'.'.join(attribute_path)}",
                    ) from exc
            if invoke_zero_arg or invoke_args or invoke_kwargs:
                if not callable(value):
                    raise FallbackTargetResolutionError(
                        "P_PYREF_RUNTIME_UNRESOLVED",
                        uid,
                        f"Python provider {root_name}.{'.'.join(attribute_path)} is not callable",
                    )
                value = value(*tuple(invoke_args), **dict(invoke_kwargs))
            return value

        clear = getattr(model, "clear_all", None)
        return cls(
            run_keys=run_keys,
            policy_factory=policy_factory,
            target_resolver=target_resolver,
            clear_policy_state=clear if callable(clear) else None,
            python_reference_resolver=python_reference_resolver,
        )

    @classmethod
    def from_fast_plan(
        cls,
        *,
        plan: Any,
        model: Any,
        space: Any,
        target_plan: FallbackTargetPlan | None = None,
    ) -> "FallbackPolicyRuntime":
        """Build the normal compiler-owned modelx fallback runtime.

        No per-UID source-name map or cross-Space resolver callback is required.
        Identity comes entirely from the static fallback target plan derived from
        the same modelx-cython metadata used by native cross-Space binding.
        """

        if target_plan is None:
            target_plan = build_fallback_target_plan(plan)
        if not target_plan.supported:
            blocker = target_plan.blockers[0]
            raise FallbackTargetResolutionError(blocker.code, blocker.uid, blocker.detail)

        def policy_factory(key: Any) -> Any:
            # Unparameterized Spaces use the singleton invocation key ``()`` in
            # ranked plans but are not itemizable. Resolve against the Space
            # itself; parameterized Spaces continue to bind their ItemSpace.
            if getattr(space, "formula", None) is None:
                return space
            return space[key]

        def bind_target_space(scope: Any, root_context: Any, hop: FallbackSpaceHop) -> Any:
            current = scope
            for attr in hop.receiver_path:
                try:
                    current = getattr(current, attr)
                except Exception as exc:
                    raise FallbackTargetResolutionError(
                        "P_TARGET_RUNTIME_UNRESOLVED",
                        "",
                        f"cannot resolve receiver attribute {attr!r} on path {hop.receiver_path!r}",
                    ) from exc
            if hop.target_space_parameters:
                # A relative reference may already yield a parameter-bound ItemSpace.
                # If it does not, use the root policy's named parameter values to
                # instantiate the metadata-declared target Space.  The frontend
                # admits cross-Space classes only when these names are a subset of
                # the root Space parameters, so there is no value guessing here.
                already_bound = all(hasattr(current, name) for name in hop.target_space_parameters)
                if not already_bound:
                    values: list[Any] = []
                    for name in hop.target_space_parameters:
                        if not hasattr(root_context, name):
                            raise FallbackTargetResolutionError(
                                "P_TARGET_PARAMETER_UNRESOLVED",
                                "",
                                f"active root ItemSpace has no parameter {name!r}",
                            )
                        values.append(getattr(root_context, name))
                    key: Any = values[0] if len(values) == 1 else tuple(values)
                    try:
                        current = current[key]
                    except Exception as exc:
                        raise FallbackTargetResolutionError(
                            "P_TARGET_PARAMETER_UNRESOLVED",
                            "",
                            f"cannot bind target Space {hop.target_class_fullname} with key {key!r}",
                        ) from exc
            return current

        def owner_scope(context: Any, uid: str) -> Any:
            spec = target_plan.target(uid)
            scope = context
            for hop in spec.hops:
                try:
                    scope = bind_target_space(scope, context, hop)
                except FallbackTargetResolutionError as exc:
                    if exc.uid:
                        raise
                    raise FallbackTargetResolutionError(exc.code, uid, exc.detail) from exc
            return scope

        def target_resolver(context: Any, uid: str, _source_name: str) -> Callable[..., Any]:
            spec = target_plan.target(uid)
            scope = owner_scope(context, uid)
            try:
                target = getattr(scope, spec.source_name)
            except Exception as exc:
                raise FallbackTargetResolutionError(
                    "P_TARGET_RUNTIME_UNRESOLVED",
                    uid,
                    f"resolved owner {spec.owner_class_fullname} has no Cell {spec.source_name!r}",
                ) from exc
            if not callable(target):
                raise FallbackTargetResolutionError(
                    "P_TARGET_RUNTIME_UNRESOLVED",
                    uid,
                    f"resolved target {spec.source_fullname} is not callable",
                )
            return target

        run_keys = tuple(getattr(plan, "run_keys", ()))

        def python_reference_resolver(
            policy_i: int,
            uid: str,
            root_name: str,
            attribute_path: tuple[str, ...],
            invoke_zero_arg: bool,
            invoke_args: tuple[Any, ...] = (),
            invoke_kwargs: tuple[tuple[str, Any], ...] = (),
        ) -> Any:
            i = int(policy_i)
            if i < 0 or i >= len(run_keys):
                raise IndexError(f"Python-reference policy index {i} outside [0,{len(run_keys)})")
            context = policy_factory(run_keys[i])
            scope = owner_scope(context, uid)
            if root_name in {"self", "cls"}:
                value = scope
            else:
                try:
                    value = getattr(scope, root_name)
                except Exception as exc:
                    raise FallbackTargetResolutionError(
                        "P_PYREF_RUNTIME_UNRESOLVED",
                        uid,
                        f"owner {target_plan.target(uid).owner_class_fullname} has no reference {root_name!r}",
                    ) from exc
            for attr in attribute_path:
                try:
                    value = getattr(value, attr)
                except Exception as exc:
                    raise FallbackTargetResolutionError(
                        "P_PYREF_RUNTIME_UNRESOLVED",
                        uid,
                        f"cannot resolve Python reference attribute {attr!r} in "
                        f"{root_name}.{'.'.join(attribute_path)}",
                    ) from exc
            if invoke_zero_arg or invoke_args or invoke_kwargs:
                if not callable(value):
                    raise FallbackTargetResolutionError(
                        "P_PYREF_RUNTIME_UNRESOLVED",
                        uid,
                        f"Python provider {root_name}.{'.'.join(attribute_path)} is not callable",
                    )
                value = value(*tuple(invoke_args), **dict(invoke_kwargs))
            return value

        clear = getattr(model, "clear_all", None)
        return cls(
            run_keys=run_keys,
            policy_factory=policy_factory,
            target_resolver=target_resolver,
            clear_policy_state=clear if callable(clear) else None,
            python_reference_resolver=python_reference_resolver,
        )

    @classmethod
    def from_callable_map(
        cls,
        *,
        run_keys: Sequence[Any],
        callables: Mapping[str, Callable[..., Any]],
        policy_factory: Callable[[Any], Any] | None = None,
        clear_policy_state: Callable[[], None] | None = None,
    ) -> "FallbackPolicyRuntime":
        """Small deterministic runtime used by unit tests and microbenchmarks."""
        callables = dict(callables)
        factory = policy_factory or (lambda key: key)

        def resolver(_context: Any, uid: str, _source_name: str) -> Callable[..., Any]:
            try:
                return callables[uid]
            except KeyError as exc:
                raise KeyError(f"no Python-backed callable bound for {uid!r}") from exc

        return cls(
            run_keys=run_keys,
            policy_factory=factory,
            target_resolver=resolver,
            clear_policy_state=clear_policy_state,
        )

    def start_batch(self) -> None:
        # Defensively release anything left by a prior interrupted batch.
        self.finish_batch()
        self._started = True

    def _discard_active_policy(self) -> None:
        if self._active_policy_i is None:
            return
        self._targets.clear()
        self._active_context = None
        if self._clear_policy_state is not None:
            self._clear_policy_state()
        self._active_policy_i = None
        self._policy_discards += 1

    def finish_policy(self) -> None:
        """Discard the active fallback context at the current policy boundary."""
        self._discard_active_policy()

    def finish_batch(self) -> None:
        self._discard_active_policy()
        self._started = False

    def _activate(self, policy_i: int) -> None:
        i = int(policy_i)
        if i < 0 or i >= len(self.run_keys):
            raise IndexError(f"fallback policy index {i} outside [0,{len(self.run_keys)})")
        if self._active_policy_i == i:
            return
        self._discard_active_policy()
        self._active_context = self._policy_factory(self.run_keys[i])
        self._active_policy_i = i
        self._targets.clear()
        self._policy_activations += 1

    def resolve_python_reference_series(
        self,
        uid: str,
        root_name: str,
        attribute_path: Sequence[str],
        invoke_zero_arg: bool,
        invoke_args: Sequence[Any] = (),
        invoke_kwargs: Sequence[tuple[str, Any]] = (),
    ) -> tuple[Any, ...]:
        """Resolve one detached Python reference for every run key.

        Resolution happens during module binding, outside recursive execution.
        Object-valued zero-argument provider Cells are therefore materialized
        before timing/projection and are never called from a P formula body.
        """
        if self._python_reference_resolver is None:
            raise MixedRecursiveNativeError(
                "detached Python reference binding requires a compiler-owned reference resolver"
            )
        attrs = tuple(str(x) for x in attribute_path)
        args = tuple(invoke_args)
        kwargs = tuple((str(name), value) for name, value in invoke_kwargs)
        invokes_provider = bool(invoke_zero_arg or args or kwargs)
        values: list[Any] = []
        for i in range(len(self.run_keys)):
            values.append(
                self._python_reference_resolver(
                    i, str(uid), str(root_name), attrs, bool(invoke_zero_arg), args, kwargs
                )
            )
            self._python_reference_resolutions += 1
            if invokes_provider:
                self._python_provider_materializations += 1
        return tuple(values)

    def invoke_native(
        self,
        uid: str,
        source_name: str,
        policy_i: int,
        result_dtype: str,
        *args: Any,
    ) -> bool | int | float:
        if not self._started:
            # Generated execution is deliberately lazy too: no backend call is
            # made until the first native memo miss reaches this method.
            self._started = True
        if result_dtype not in _NATIVE_DTYPES:
            raise MixedRecursiveNativeError(f"{uid}: non-native fallback result ABI {result_dtype!r}")
        self._activate(policy_i)
        target = self._targets.get(uid)
        if target is None:
            target = self._target_resolver(self._active_context, uid, source_name)
            self._targets[uid] = target
        self._python_invocations += 1
        self._per_cell[uid] += 1
        return _convert_native_result(target(*args), result_dtype)

    def stats(self) -> dict[str, Any]:
        return {
            "python_invocations": self._python_invocations,
            "policy_activations": self._policy_activations,
            "policy_discards": self._policy_discards,
            "active_policy_i": self._active_policy_i,
            "active_target_count": len(self._targets),
            "per_cell_python_invocations": dict(sorted(self._per_cell.items())),
            "python_reference_resolutions": self._python_reference_resolutions,
            "python_provider_materializations": self._python_provider_materializations,
        }

    def reset_stats(self) -> None:
        self._python_invocations = 0
        self._policy_activations = 0
        self._policy_discards = 0
        self._per_cell.clear()
        self._python_reference_resolutions = 0
        self._python_provider_materializations = 0


@dataclass(frozen=True)
class MixedRecursiveProgram:
    schema: str
    formula_program: NativeFormulaProgram
    compile_plan: SourcePreservingCompilePlan
    dense_program: DenseRecursiveProgram
    execution_class_plan: ExecutionClassPlan
    object_memo_plan: ObjectMemoPlan
    python_key_memo_plan: PythonKeyMemoPlan
    fallback_target_plan: FallbackTargetPlan
    python_backed_uids: tuple[str, ...]
    object_backed_uids: tuple[str, ...]
    ranked_backed_uids: tuple[str, ...]
    gil_backed_uids: tuple[str, ...]
    modelx_fallback_uids: tuple[str, ...]
    gil_backed_cells: Mapping[str, NativeFormulaCell]
    python_key_cells: Mapping[str, NativeFormulaCell]
    python_key_backed_uids: tuple[str, ...]
    supported: bool
    python_output_uid: str | None = None
    python_output_shape: tuple[int, ...] | None = None
    python_output_dtype: str | None = None
    core_cache_cython_p_uids: tuple[str, ...] = ()
    category_bridge_domain_ids: tuple[str, ...] = ()
    python_reference_specs: tuple[PythonReferenceSpec, ...] = ()
    blockers: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "supported": self.supported,
            "blockers": list(self.blockers),
            "output_uid": self.execution_class_plan.output_uid,
            "native_driver_output_uid": self.formula_program.output_uid,
            "python_output_uid": self.python_output_uid,
            "python_output_shape": (None if self.python_output_shape is None else list(self.python_output_shape)),
            "python_output_dtype": self.python_output_dtype,
            "recursive_abi": "r_<cell>_native_key_for_N0_N1_bridgeable_P_plus_python_key_for_detached_helpers",
            "python_entry": "calc_<P>_after_native_memo_miss_then_python_key_r_<helper>_under_GIL",
            "python_backed_uids": list(self.python_backed_uids),
            "object_backed_uids": list(self.object_backed_uids),
            "ranked_backed_uids": list(self.ranked_backed_uids),
            "object_memo_plan": self.object_memo_plan.manifest(),
            "python_key_memo_plan": self.python_key_memo_plan.manifest(),
            "gil_backed_uids": list(self.gil_backed_uids),
            "python_key_backed_uids": list(self.python_key_backed_uids),
            "core_cache_cython_p_uids": list(self.core_cache_cython_p_uids),
            "category_bridge_domain_ids": list(self.category_bridge_domain_ids),
            "python_reference_specs": [x.manifest() for x in self.python_reference_specs],
            "modelx_fallback_uids": list(self.modelx_fallback_uids),
            "expression_level_python_islands": bool(
                self.gil_backed_uids or self.python_key_backed_uids
            ),
            "python_semantic_execution": "detached_generated_Cython_formula_with_native_r_and_python_key_r_dependencies_and_bound_python_refs",
            "live_modelx_formula_execution": bool(self.modelx_fallback_uids),
            "python_dict_cache_for_native_abi_cells": False,
            "python_dict_cache_for_python_key_helpers": bool(self.python_key_backed_uids),
            "dense_program": self.dense_program.manifest(),
            "execution_class_plan": self.execution_class_plan.manifest(),
            "fallback_target_plan": self.fallback_target_plan.manifest(),
        }


def _typed_zero(dtype: str) -> ast.Constant:
    if dtype == "float64":
        return ast.Constant(0.0)
    if dtype in {"int64", "bool"}:
        return ast.Constant(0)
    raise MixedRecursiveNativeError(dtype)


def _python_root_driver_cell(existing_uids: set[str]) -> NativeFormulaCell:
    """Return a tiny scalar native root for a selected Python-object output.

    The selected P Cell itself executes under the GIL through the compiler-owned
    object memo.  The synthetic root exists only to keep the native dependency
    closure on the ordinary scalar/per-policy recursive generator.  This is
    important: the ranked generator has no policy index and is therefore not a
    valid host for point-scoped inputs used by a Python/DataFrame result Cell.
    ``run_all`` is rewritten later to evaluate the selected P root once per
    policy while advancing the normal native memo epoch.
    """
    uid = "__mx_python_root_driver"
    suffix = 0
    while uid in existing_uids:
        suffix += 1
        uid = f"__mx_python_root_driver_{suffix}"
    fn = ast.FunctionDef(
        name=uid,
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[ast.Return(ast.Constant(0.0))],
        decorator_list=[],
    )
    ast.fix_missing_locations(fn)
    return NativeFormulaCell(
        uid=uid,
        source_name=uid,
        source_fullname=uid,
        dtype="float64",
        parameters=(),
        parameter_dtypes=(),
        function=copy.deepcopy(fn),
        reduction=None,
        logical_result_ndim=0,
        logical_result_shape=(),
        ranked_function=None,
    )


def _stub_python_backed_body(cell: NativeFormulaCell) -> NativeFormulaCell:
    """Hide P syntax from native Cython ABI analysis; body is never emitted."""
    if cell.function is None:
        fn = ast.FunctionDef(
            name=cell.uid,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=x) for x in cell.parameters],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[ast.Return(_typed_zero(cell.dtype))],
            decorator_list=[],
        )
    else:
        fn = copy.deepcopy(cell.function)
        fn.body = [ast.Return(_typed_zero(cell.dtype))]
    ast.fix_missing_locations(fn)
    return replace(cell, function=fn, reduction=None, ranked_function=None)




def _mixed_recursive_uids(
    plan: Any, catalog: Mapping[str, NativeFormulaCell]
) -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...],
    Mapping[str, NativeFormulaCell], Mapping[str, NativeFormulaCell],
    tuple[str, ...], tuple[PythonReferenceSpec, ...],
]:
    """Return native recursive closure plus detached Python-key helper closure.

    Part 2A keeps Python-key helpers physically outside the native dense memo
    program. They may be entered only from an already GIL-held detached P body.
    Native N0/N1 -> Python-key edges fail closed.
    """
    classes = plan.execution_class_plan

    (
        python_key_candidates,
        python_key_candidate_domains,
        python_key_candidate_refs,
    ) = _python_key_helper_candidates(plan, catalog)

    core_cache_cells: dict[str, NativeFormulaCell] = {}
    candidate_domains: dict[str, tuple[str, ...]] = {}
    candidate_refs: dict[str, tuple[PythonReferenceSpec, ...]] = {}
    for uid, row in classes.cells.items():
        if row.execution_class != P:
            continue
        candidate = _core_cache_cython_p_candidate(
            plan,
            uid,
            catalog,
            python_key_catalog=python_key_candidates,
        )
        if candidate is None:
            continue
        cell, domains, refs = candidate
        core_cache_cells[uid] = cell
        candidate_domains[uid] = tuple(domains)
        candidate_refs[uid] = tuple(refs)

    gil_candidates = set(classes.gil_cython_candidate_uids) | set(core_cache_cells)
    native_seen: set[str] = set()
    helper_seen: set[str] = set()
    unsupported: set[str] = set()
    pending_native = [classes.output_uid]
    pending_helpers: list[str] = []

    while pending_native or pending_helpers:
        while pending_helpers:
            uid = pending_helpers.pop()
            if uid in helper_seen:
                continue
            helper = python_key_candidates.get(uid)
            if helper is None:
                unsupported.add(uid)
                continue
            helper_seen.add(uid)
            deps = {
                node.func.id
                for node in ast.walk(helper.function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in catalog
            }
            for dep in sorted(deps):
                if dep in python_key_candidates:
                    pending_helpers.append(dep)
                else:
                    pending_native.append(dep)

        if not pending_native:
            continue
        uid = pending_native.pop()
        if uid in native_seen or uid not in classes.cells:
            continue
        row = classes.cell(uid)
        native_seen.add(uid)
        if row.execution_class == P and uid not in gil_candidates:
            unsupported.add(uid)
            continue
        if uid in core_cache_cells:
            deps = {
                node.func.id
                for node in ast.walk(core_cache_cells[uid].function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in catalog
            }
        else:
            deps = set(row.dependencies)
        for dep in sorted(deps):
            if dep in python_key_candidates:
                if row.execution_class != P:
                    raise MixedRecursiveNativeError(
                        f"{uid}:{row.source_name}: direct native {row.execution_class} -> "
                        f"Python-key helper {dep}:{classes.cell(dep).source_name} is not allowed"
                    )
                pending_helpers.append(dep)
            else:
                pending_native.append(dep)

    recursive = tuple(sorted(native_seen))
    recursive_set = set(recursive)
    gil_backed = tuple(sorted(uid for uid in recursive if uid in gil_candidates))
    unsupported_python = tuple(sorted(unsupported))
    selected_python_key = {
        uid: python_key_candidates[uid]
        for uid in sorted(helper_seen)
        if uid in python_key_candidates
    }

    bridge_domains: set[str] = set()
    refs_by_var: dict[str, PythonReferenceSpec] = {}
    reference_sources: list[tuple[str, tuple[PythonReferenceSpec, ...]]] = []
    for uid in sorted(recursive_set & set(core_cache_cells)):
        bridge_domains.update(candidate_domains.get(uid, ()))
        reference_sources.append((uid, candidate_refs.get(uid, ())))
    for uid in sorted(selected_python_key):
        bridge_domains.update(python_key_candidate_domains.get(uid, ()))
        reference_sources.append((uid, python_key_candidate_refs.get(uid, ())))

    for _uid, specs in reference_sources:
        for spec in specs:
            previous = refs_by_var.get(spec.variable_name)
            if previous is not None and previous != spec:
                raise MixedRecursiveNativeError(
                    f"conflicting detached Python reference variable {spec.variable_name!r}"
                )
            refs_by_var[spec.variable_name] = spec

    return (
        recursive,
        gil_backed,
        unsupported_python,
        core_cache_cells,
        selected_python_key,
        tuple(sorted(bridge_domains)),
        tuple(sorted(refs_by_var.values(), key=lambda x: x.variable_name)),
    )



def _late_native_admission_reclassification(
    plan: Any,
    compile_plan: SourcePreservingCompilePlan,
) -> tuple[str, ...]:
    """Demote native candidates that fail the final source-preserving ABI gate.

    Execution classification is intentionally optimistic before the whole mixed
    closure is assembled: N0/N1 means "native candidate", not "already proved by
    the final source-preserving ABI analysis".  A candidate can therefore fail a
    later whole-body check (for example an unproved generator-expression return
    dtype).  Such a failure is intrinsic to that Cell body and must make *that
    Cell* P; it must not abort the entire mixed program or force its native
    callers out of the recursive ABI.

    This routine records the late blocker as ordinary direct classification
    evidence and rebuilds the graph monotonically.  Callers then become N1 by
    dependency propagation, exactly as they would have if the blocker had been
    visible during the first classification pass.
    """
    classes = getattr(plan, "execution_class_plan", None)
    catalog = dict(getattr(plan, "source_transparent_cell_catalog", {}))
    if classes is None or not catalog:
        return ()

    direct: dict[str, list[tuple[str, str]]] = {
        uid: list(classes.cell(uid).direct_blockers)
        for uid in catalog
        if uid in classes.cells
    }
    demoted: list[str] = []
    for raw in compile_plan.blockers:
        uid, sep, detail = str(raw).partition(":")
        if not sep or uid not in classes.cells or uid not in catalog:
            continue
        row = classes.cell(uid)
        if row.execution_role != "formula" or row.execution_class == P:
            continue
        if not row.native_scalar_abi or not row.bridgeable_python_fallback:
            continue
        blocker = ("native_admission", detail)
        bucket = direct.setdefault(uid, list(row.direct_blockers))
        if blocker in bucket:
            continue
        bucket.append(blocker)
        demoted.append(uid)

    if not demoted:
        return ()

    frozen_direct = {
        uid: tuple(sorted(set(items)))
        for uid, items in direct.items()
    }
    updated = build_execution_class_plan(
        output_uid=classes.output_uid,
        cells=catalog,
        direct_blockers=frozen_direct,
        parameter_ranges=getattr(plan, "parameter_ranges", {}),
        native_hash_capacity_by_uid=getattr(plan, "native_hash_capacity_by_uid", {}),
    )
    plan.execution_class_plan = updated
    # Keep the plan's direct-blocker authority synchronized so a later consumer
    # cannot accidentally reconstruct the pre-demotion classification.
    try:
        plan.source_transparent_direct_blockers = frozen_direct
    except Exception:
        pass
    return tuple(sorted(demoted))


def _validate_category_completion(plan) -> None:
    """Reject incomplete category ABIs before emitting native/Python bridges.

    Demoting a caller to Python does not repair a native callee whose cache and
    branch lowering exclude one of the caller's reachable labels.
    """
    categories = getattr(plan, "category_plan", None)
    if categories is None:
        return
    for target, proof in categories.parameter_domain_completion_proofs.items():
        if proof.get("rejected"):
            raise MixedRecursiveNativeError(
                f"incomplete category ABI for {target!r}: {proof['rejected']!r}; "
                "cannot safely bridge a Python caller to the narrowed native callee"
            )
    for blocker in categories.encoding.blockers:
        if blocker.startswith("selected_category_parameter_completion:"):
            raise MixedRecursiveNativeError(blocker)


def build_mixed_recursive_program_from_fast_plan(
    plan: Any,
    *,
    max_dense_slots_per_cell: int = 1_000_000,
    native_hash_capacity_by_uid: Mapping[str, int] | None = None,
    native_hash_debug: bool = False,
) -> MixedRecursiveProgram:
    """Physicalize mixed recursion with native and detached Python P paths."""
    _validate_category_completion(plan)
    classes = getattr(plan, "execution_class_plan", None)
    if classes is None:
        raise MixedRecursiveNativeError("fast plan has no N0/N1/P execution-class plan")
    if classes.output_class not in {N0, N1, P}:
        raise MixedRecursiveNativeError(
            f"selected output {classes.output_uid!r} has unsupported execution class {classes.output_class}"
        )
    if classes.output_class == P and not classes.cell(classes.output_uid).bridgeable_python_fallback:
        raise MixedRecursiveNativeError(
            f"selected P output {classes.output_uid!r} has no compiler-owned recursive value ABI"
        )

    catalog = dict(getattr(plan, "source_transparent_cell_catalog", {}))
    (
        recursive_uids, gil_backed_uids, unsupported_python_uids,
        core_cache_cython_cells, python_key_cells,
        category_bridge_domain_ids, python_reference_specs,
    ) = _mixed_recursive_uids(plan, catalog)
    if not recursive_uids:
        raise MixedRecursiveNativeError("selected output has no recursive ABI closure")
    missing = sorted(set(recursive_uids) - set(catalog))
    if missing:
        raise MixedRecursiveNativeError("missing source-transparent Cells: " + ",".join(missing))

    if unsupported_python_uids:
        rendered = ",".join(
            f"{uid}:{classes.cell(uid).source_name}" for uid in unsupported_python_uids
        )
        raise MixedRecursiveNativeError(
            "detached Python P source unavailable; live modelx formula execution is disabled: "
            + rendered
        )

    python_p_uids = tuple(
        uid for uid in recursive_uids if classes.cell(uid).execution_class == P
    )
    python_key_uids = tuple(sorted(python_key_cells))
    reference_owner_uids = tuple(sorted({spec.owner_uid for spec in python_reference_specs}))
    # Owner-space identity is needed for detached P bodies and for any ranked/
    # native formula whose remaining Python references are prebound. The owner
    # Cell itself is never invoked through modelx.
    fallback_targets = build_fallback_target_plan(
        plan, p_uids=tuple(sorted(
            set(python_p_uids) | set(python_key_uids) | set(reference_owner_uids)
        ))
    )
    if not fallback_targets.supported:
        blocker = fallback_targets.blockers[0]
        raise FallbackTargetResolutionError(blocker.code, blocker.uid, blocker.detail)

    gil_backed_set = set(gil_backed_uids)
    modelx_fallback_uids: tuple[str, ...] = ()
    modelx_fallback_set: set[str] = set()
    cells: dict[str, NativeFormulaCell] = {}
    python_backed: list[str] = []
    object_backed: list[str] = []
    ranked_backed: list[str] = []
    gil_cells: dict[str, NativeFormulaCell] = {}
    for uid in recursive_uids:
        cell = catalog[uid]
        row = classes.cell(uid)
        if cell.execution_role != "formula":
            cells[uid] = cell
            continue
        if row.execution_class == P:
            if not row.bridgeable_python_fallback:
                raise MixedRecursiveNativeError(f"{uid}: P boundary lacks a recursive value ABI")
            python_backed.append(uid)
            if uid in gil_backed_set:
                gil_cells[uid] = core_cache_cython_cells.get(uid, cell)
            else:
                raise MixedRecursiveNativeError(
                    f"{uid}: P Cell has no detached Python-semantic execution path"
                )
            if row.value_abi == PYTHON_OBJECT_VALUE_ABI:
                object_backed.append(uid)
            elif row.value_abi == RANKED_ARRAY_VALUE_ABI:
                object_backed.append(uid)
                ranked_backed.append(uid)
            elif row.value_abi == NATIVE_SCALAR_VALUE_ABI:
                cells[uid] = _stub_python_backed_body(cell)
            else:
                raise MixedRecursiveNativeError(f"{uid}: unsupported P result ABI {row.value_abi!r}")
        else:
            cells[uid] = cell

    python_output_uid: str | None = None
    python_output_shape: tuple[int, ...] | None = None
    python_output_dtype: str | None = None
    native_driver_output_uid = classes.output_uid
    if classes.output_class == P:
        output_row = classes.cell(classes.output_uid)
        output_cell = catalog[classes.output_uid]
        if output_row.value_abi in {PYTHON_OBJECT_VALUE_ABI, RANKED_ARRAY_VALUE_ABI}:
            if output_cell.logical_result_shape is None:
                raise MixedRecursiveNativeError(
                    f"selected Python object output {classes.output_uid!r} requires fixed result geometry"
                )
            python_output_uid = classes.output_uid
            python_output_shape = tuple(int(x) for x in output_cell.logical_result_shape)
            # A Python list/tuple keeps the object ABI internally, but public
            # run_all converts it to ndarray. Use observed conversion geometry
            # for that boundary only; never reinterpret list formulas as arrays.
            module_info = getattr(plan, 'module_info', None)
            runtime_info = (getattr(getattr(module_info, 'logger', None), 'cells_info', {})
                            .get(output_cell.source_fullname))
            sequence_shape = getattr(runtime_info, 'numeric_sequence_shape', None)
            if sequence_shape is not None:
                python_output_shape = tuple(int(x) for x in sequence_shape)
            python_output_dtype = output_cell.dtype
            driver = _python_root_driver_cell(set(cells) | set(catalog))
            cells[driver.uid] = driver
            native_driver_output_uid = driver.uid

    # GIL-backed P formulas consume the same prepared native inputs as N0/N1.
    # Only whole-Cell modelx fallbacks omit their provider inputs from this module.
    from .modelx_cython_fast_backend import _freeze_input, _input_keys

    runtime_specs = dict(getattr(plan, "source_preserving_runtime_inputs", {}) or getattr(plan, "runtime_inputs", {}))
    used_input_keys: set[str] = set()
    for uid in recursive_uids:
        if uid in modelx_fallback_set:
            continue
        used_input_keys.update(_input_keys(catalog[uid]))
    for uid in python_key_uids:
        used_input_keys.update(_input_keys(catalog[uid]))
    for key in tuple(used_input_keys):
        spec = runtime_specs.get(key)
        if spec is not None and getattr(spec, "category_validity_key", None):
            used_input_keys.add(spec.category_validity_key)
    unknown_inputs = sorted(used_input_keys - set(runtime_specs))
    if unknown_inputs:
        raise MixedRecursiveNativeError("missing prepared inputs: " + ",".join(unknown_inputs))
    selected_specs = {key: runtime_specs[key] for key in runtime_specs if key in used_input_keys}
    if not any(getattr(spec, "scope", None) == "point" for spec in selected_specs.values()):
        for key, spec in runtime_specs.items():
            if getattr(spec, "scope", None) == "point":
                selected_specs[key] = spec
                break
    frozen_inputs = tuple(_freeze_input(spec) for spec in selected_specs.values())

    module_refs = tuple(sorted(dict(getattr(plan, "source_transparent_module_refs", {})).items()))
    if not module_refs:
        base = getattr(plan, "source_preserving_native_program", None) or getattr(plan, "native_program", None)
        if base is not None:
            module_refs = tuple(base.module_refs)

    category_domains: tuple[tuple[str, tuple[object, ...]], ...] = ()
    category_plan = getattr(plan, "category_plan", None)
    if category_plan is not None:
        try:
            category_domains = tuple(
                (did, tuple(domain.labels))
                for did, domain in sorted(category_plan.encoding.domains.items())
            )
        except AttributeError:
            category_domains = ()

    formula = NativeFormulaProgram(
        schema="modelx_graph.native_formula.mixed_recursive_dev131.v1",
        uid=str(getattr(plan, "schema", "fast_plan")) + "_dev131",
        output_uid=native_driver_output_uid,
        cells=cells,
        input_order=tuple(selected_specs),
        inputs=frozen_inputs,
        normalized_operators=tuple(getattr(plan, "normalized_operators", ())),
        module_refs=module_refs,
        provenance="architecture_b_dev131_gil_cython_python_semantics",
        category_domains=category_domains,
    )

    bound_inputs: dict[str, Any] = {}
    if frozen_inputs:
        all_bound = plan.bind_source_preserving_inputs()
        bound_inputs = {key: all_bound[key] for key in selected_specs}
    input_bounds = derive_complete_input_numeric_bounds(
        formula, bound_inputs, run_population=len(plan.run_keys),
    )
    lexical = {
        uid: source
        for uid, source in dict(getattr(plan, "formula_lexical_sources", {})).items()
        if uid in cells and uid in classes.cells and classes.cell(uid).execution_class != P
    }
    compile_plan = build_source_preserving_compile_plan(
        formula, input_bounds=input_bounds, lexical_sources=lexical,
    )
    if not compile_plan.supported:
        demoted = _late_native_admission_reclassification(plan, compile_plan)
        if demoted:
            # Rebuild from the now-authoritative graph.  Every recursive pass
            # adds at least one direct P blocker, so this is a finite monotone
            # classification fixed point rather than retry-on-error behavior.
            return build_mixed_recursive_program_from_fast_plan(
                plan,
                max_dense_slots_per_cell=max_dense_slots_per_cell,
                native_hash_capacity_by_uid=native_hash_capacity_by_uid,
                native_hash_debug=native_hash_debug,
            )
        raise MixedRecursiveNativeError(
            "mixed native admission blocked after P reclassification: "
            + "; ".join(compile_plan.blockers)
        )

    capacities = (
        dict(getattr(plan, "native_hash_capacity_by_uid", {}))
        if native_hash_capacity_by_uid is None
        else dict(native_hash_capacity_by_uid)
    )
    dense = build_dense_recursive_program(
        compile_plan.formula_program,
        parameter_ranges=getattr(plan, "parameter_ranges", {}),
        max_dense_slots_per_cell=max_dense_slots_per_cell,
        native_hash_capacity_by_uid=capacities,
        native_hash_debug=native_hash_debug,
    )
    if not dense.supported:
        raise MixedRecursiveNativeError("dev131 native memo plan blocked: " + "; ".join(dense.blockers))
    for uid in python_backed:
        if uid in object_backed:
            continue
        memo = dense.memo_plan.cell(uid)
        if memo.kind not in {"scalar", "dense", "hash"}:
            raise MixedRecursiveNativeError(
                f"{uid}: Python-backed native ABI Cell must use native scalar/dense/hash memo, got {memo.kind}"
            )

    object_cells = {uid: gil_cells[uid] for uid in object_backed}
    object_memo = build_object_memo_plan(
        object_cells,
        parameter_ranges=getattr(plan, "parameter_ranges", {}),
        native_hash_capacity_by_uid=capacities,
        max_dense_slots_per_cell=max_dense_slots_per_cell,
    )
    python_key_memo = build_python_key_memo_plan(
        python_key_cells, selected_uids=python_key_uids
    )

    return MixedRecursiveProgram(
        schema="modelx_graph.mixed_recursive_program.v5_python_key_helpers_part2a",
        formula_program=compile_plan.formula_program,
        compile_plan=compile_plan,
        dense_program=dense,
        execution_class_plan=classes,
        object_memo_plan=object_memo,
        python_key_memo_plan=python_key_memo,
        fallback_target_plan=fallback_targets,
        python_backed_uids=tuple(sorted(python_backed)),
        object_backed_uids=tuple(sorted(object_backed)),
        ranked_backed_uids=tuple(sorted(ranked_backed)),
        gil_backed_uids=gil_backed_uids,
        modelx_fallback_uids=modelx_fallback_uids,
        gil_backed_cells=gil_cells,
        python_key_cells=dict(python_key_cells),
        python_key_backed_uids=python_key_uids,
        python_output_uid=python_output_uid,
        python_output_shape=python_output_shape,
        python_output_dtype=python_output_dtype,
        core_cache_cython_p_uids=tuple(sorted(core_cache_cython_cells)),
        category_bridge_domain_ids=category_bridge_domain_ids,
        python_reference_specs=python_reference_specs,
        supported=True,
    )


class MixedRecursiveCythonGenerator(SourcePreservingDenseCythonGenerator):
    """Cython generator where P differs only in ``calc_<P>`` cache-miss body."""

    def __init__(
        self,
        program: MixedRecursiveProgram,
        *,
        diagnostic_instrumentation: bool = False,
        openmp: bool = False,
    ):
        if not program.supported:
            raise MixedRecursiveNativeError("mixed recursive program is blocked")
        self.mixed_program = program
        self.compile_plan = program.compile_plan
        self.diagnostic_instrumentation = bool(diagnostic_instrumentation)
        self.openmp = bool(openmp)
        if self.openmp and (
            program.python_backed_uids
            or program.python_key_backed_uids
            or diagnostic_instrumentation
        ):
            raise MixedRecursiveNativeError("OpenMP requires a pure native closure without diagnostic instrumentation")
        # Skip SourcePreservingDenseCythonGenerator's wrapper type check; this
        # object intentionally carries a richer Stage-2B program wrapper.
        DenseRecursiveCythonGenerator.__init__(self, program.dense_program)
        self.object_cells = {
            uid: program.gil_backed_cells[uid]
            for uid in program.object_backed_uids
        }
        self.object_memo_emitter = ObjectMemoCythonEmitter(
            program.object_memo_plan, self.object_cells
        )
        self.python_key_cells = dict(program.python_key_cells)
        self.python_key_memo_emitter = PythonKeyMemoCythonEmitter(
            program.python_key_memo_plan, self.python_key_cells
        )
        self.has_native_round = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "round"
            and 1 <= len(node.args) <= 2
            for uid, cell in self.v.items()
            if cell.function is not None
            for node in ast.walk(cell.function)
        )

    @property
    def _ranked_program(self) -> bool:
        output = self.v[self.formula.output_uid]
        return output.logical_result_ndim == 1 and output.ranked_function is not None

    def _ranked_function_source(self, cell: NativeFormulaCell) -> str:
        """Render ranked formulas with object calls carrying rank-policy 0.

        A ranked invocation evaluates the whole point axis at once, so its
        modelx invocation domain is the singleton run key.  Object-memo calls
        therefore use policy index zero while their explicit formula arguments
        remain the native memo key.
        """
        rendered = cell
        row = (
            self.mixed_program.execution_class_plan.cell(cell.uid)
            if cell.uid in self.mixed_program.execution_class_plan.cells
            else None
        )
        if row is not None and row.execution_class == P:
            rendered = self.mixed_program.gil_backed_cells[cell.uid]
        fn = rendered.ranked_function if rendered.ranked_function is not None else rendered.function
        if fn is None:
            raise MixedRecursiveNativeError(f"{cell.uid}: ranked formula is missing source")
        transformed = _PythonKeyMemoCallRewriter(
            self.python_key_cells, ast.Constant(0)
        ).visit(copy.deepcopy(fn))
        transformed = _ObjectMemoCallRewriter(
            self.object_cells, ast.Constant(0)
        ).visit(transformed)
        ref_vars = {
            spec.variable_name
            for spec in self.mixed_program.python_reference_specs
            if spec.owner_uid == cell.uid
        }
        if ref_vars:
            transformed = _PolicyIndexedPythonReferenceRewriter(
                ref_vars, ast.Constant(0)
            ).visit(transformed)
        assert isinstance(transformed, ast.FunctionDef)
        ast.fix_missing_locations(transformed)
        if rendered.ranked_function is not None:
            rendered = replace(rendered, ranked_function=transformed)
        else:
            rendered = replace(rendered, function=transformed)
        return DenseRecursiveCythonGenerator._ranked_function_source(self, rendered)

    def _diag_names(self, uid: str) -> tuple[str, str, str]:
        safe = self._safe_uid(uid)
        return (
            f"_diag_r_calls_{safe}",
            f"_diag_memo_hits_{safe}",
            f"_diag_memo_misses_{safe}",
        )

    @staticmethod
    def _insert_after_line(lines: list[str], needle: str, new_lines: list[str]) -> list[str]:
        try:
            idx = lines.index(needle)
        except ValueError as exc:
            raise MixedRecursiveNativeError(f"diagnostic codegen anchor not found: {needle}") from exc
        return [*lines[: idx + 1], *new_lines, *lines[idx + 1 :]]

    @staticmethod
    def _insert_before_line(lines: list[str], needle: str, new_lines: list[str]) -> list[str]:
        try:
            idx = lines.index(needle)
        except ValueError as exc:
            raise MixedRecursiveNativeError(f"diagnostic codegen anchor not found: {needle}") from exc
        return [*lines[:idx], *new_lines, *lines[idx:]]

    def _emit_hash_wrapper(self, cell: NativeFormulaCell, memo: DenseMemoCell) -> list[str]:
        lines = DenseRecursiveCythonGenerator._emit_hash_wrapper(self, cell, memo)
        if not self.diagnostic_instrumentation:
            return lines
        calls, hits, misses = self._diag_names(cell.uid)
        global_idx = next((j for j, line in enumerate(lines) if line.startswith("    global ")), None)
        if global_idx is None:
            raise MixedRecursiveNativeError(f"{cell.uid}: hash diagnostic global anchor missing")
        lines[global_idx] = lines[global_idx] + f", {calls}, {hits}, {misses}"
        lines.insert(global_idx + 1, f"    {calls} += 1")
        callargs = ", ".join(cell.parameters)
        suffix = "" if not callargs else ", " + callargs
        policy_i = self._policy_index_name(cell)
        values = self._hash_value_name(cell.uid)
        lines = self._insert_before_line(
            lines,
            f"            val = calc_{cell.uid}({policy_i}{suffix})",
            [f"            {misses} += 1"],
        )
        lines = self._insert_before_line(
            lines,
            f"            return <{self.ctype(cell.dtype)}>{values}[pos]",
            [f"            {hits} += 1"],
        )
        return lines

    def _emit_wrapper(self, cell: NativeFormulaCell) -> list[str]:
        memo = self.program.memo_plan.cell(cell.uid)
        lines = DenseRecursiveCythonGenerator._emit_wrapper(self, cell)
        if not self.diagnostic_instrumentation:
            return lines
        # Hash wrappers are instrumented by _emit_hash_wrapper, which the base
        # dispatcher above calls polymorphically.
        if memo.kind == "hash":
            return lines
        calls, hits, misses = self._diag_names(cell.uid)
        if memo.kind == "direct":
            return [lines[0], f"    global {calls}, {hits}, {misses}", f"    {calls} += 1", *lines[1:]]
        if memo.kind == "dict":
            return [lines[0], f"    global {calls}, {hits}, {misses}", f"    {calls} += 1", *lines[1:]]
        global_idx = next((j for j, line in enumerate(lines) if line.startswith("    global ")), None)
        if global_idx is None:
            raise MixedRecursiveNativeError(f"{cell.uid}: memo diagnostic global anchor missing")
        lines[global_idx] = lines[global_idx] + f", {calls}, {hits}, {misses}"
        lines.insert(global_idx + 1, f"    {calls} += 1")
        values = self._arena_value_name(cell.dtype)
        lines = self._insert_before_line(
            lines,
            f"        return <{self.ctype(cell.dtype)}>{values}[idx]",
            [f"        {hits} += 1"],
        )
        callargs = ", ".join(cell.parameters)
        suffix = "" if not callargs else ", " + callargs
        policy_i = self._policy_index_name(cell)
        lines = self._insert_before_line(
            lines,
            f"    val = calc_{cell.uid}({policy_i}{suffix})",
            [f"    {misses} += 1"],
        )
        return lines

    def _diagnostic_declarations(self) -> str:
        if not self.diagnostic_instrumentation:
            return ""
        lines: list[str] = []
        for uid in sorted(self.v):
            calls, hits, misses = self._diag_names(uid)
            lines += [
                f"cdef unsigned long long {calls} = 0",
                f"cdef unsigned long long {hits} = 0",
                f"cdef unsigned long long {misses} = 0",
            ]
        return "\n".join(lines) + ("\n" if lines else "")

    def _diagnostic_helpers(self) -> str:
        if not self.diagnostic_instrumentation:
            return ""
        lines = ["cpdef dict recursive_stats():", "    return {"]
        for uid in sorted(self.v):
            calls, hits, misses = self._diag_names(uid)
            memo = self.program.memo_plan.cell(uid)
            lines.append(
                f"        {uid!r}: {{'memo_kind': {memo.kind!r}, 'r_calls': int({calls}), "
                f"'memo_hits': int({hits}), 'memo_misses': int({misses})}},"
            )
        lines += ["    }", "", "cpdef reset_recursive_stats():"]
        for uid in sorted(self.v):
            calls, hits, misses = self._diag_names(uid)
            lines += [
                f"    global {calls}, {hits}, {misses}",
                f"    {calls} = 0",
                f"    {hits} = 0",
                f"    {misses} = 0",
            ]
        lines += ["    return None", ""]
        return "\n".join(lines) + "\n"

    def _emit_gil_backed_formula(self, cell: NativeFormulaCell) -> list[str]:
        original = self.mixed_program.gil_backed_cells[cell.uid]
        fn = original.function
        if fn is None:
            raise MixedRecursiveNativeError(f"{cell.uid}: GIL-backed formula is missing source")
        policy_index_name = self._policy_index_name(original)
        transformed = copy.deepcopy(fn)
        transformed = _PythonKeyMemoCallRewriter(
            self.python_key_cells, policy_index_name
        ).visit(transformed)
        transformed = _ObjectMemoCallRewriter(
            self.object_cells, policy_index_name
        ).visit(transformed)
        transformed = _ModelxLeafRewriter(self, original, policy_index_name).visit(transformed)
        ref_vars = {
            spec.variable_name
            for spec in self.mixed_program.python_reference_specs
            if spec.owner_uid == cell.uid
        }
        if ref_vars:
            transformed = _PolicyIndexedPythonReferenceRewriter(
                ref_vars, policy_index_name
            ).visit(transformed)
        assert isinstance(transformed, ast.FunctionDef)
        ast.fix_missing_locations(transformed)
        body = list(transformed.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        body = [
            statement
            for statement in body
            if not (
                isinstance(statement, ast.Import)
                and any(alias.name == "cython" for alias in statement.names)
            )
            and not (
                isinstance(statement, ast.ImportFrom)
                and statement.module == "cython"
            )
        ]
        if not body:
            raise MixedRecursiveNativeError(f"{cell.uid}: empty GIL-backed formula body")

        parameter_types = dict(zip(original.parameters, original.parameter_dtypes))
        if cell.uid in self.mixed_program.core_cache_cython_p_uids:
            # The whole point of this tier is to let Cython own ordinary Python
            # local semantics.  Do not reuse native-path scalar local inference:
            # a local may intentionally hold a Python category label/object.
            local_dtypes = {}
        else:
            local_dtypes = dict(original.local_scalar_dtypes)
            try:
                local_dtypes.update(infer_local_types(self, fn, parameter_types))
            except Exception:
                # Local typing is an optimization here, not semantic authority. Cython
                # may keep an otherwise unknown local as a Python object under the GIL.
                pass

        ct = self.ctype(cell.dtype)
        core_cache_python = cell.uid in self.mixed_program.core_cache_cython_p_uids
        parameter_domains = tuple(original.parameter_category_domains)
        if len(parameter_domains) < len(original.parameters):
            parameter_domains = parameter_domains + (None,) * (
                len(original.parameters) - len(parameter_domains)
            )
        args_decl = [f"Py_ssize_t {policy_index_name}"]
        category_argument_prelude: list[str] = []
        for name, dtype, did in zip(
            original.parameters, original.parameter_dtypes, parameter_domains
        ):
            if core_cache_python and did is not None:
                code_name = f"__mx_catcode_{name}"
                args_decl.append(f"{self.ctype(dtype)} {code_name}")
                category_argument_prelude.append(
                    f"    {name} = {_category_helper_name('decode', did)}({code_name})"
                )
            else:
                args_decl.append(f"{self.ctype(dtype)} {name}")
        pycalc = f"pycalc_{cell.uid}"
        pycalc_ct = "object" if (
            core_cache_python and original.result_category_domain is not None
        ) else ct
        lines = [
            "@cython.boundscheck(True)",
            "@cython.wraparound(True)",
            f"cdef inline {pycalc_ct} {pycalc}({', '.join(args_decl)}):",
            "    global _guard_error",
            *category_argument_prelude,
        ]
        for name, dtype in sorted(local_dtypes.items()):
            if name in original.parameters or name == policy_index_name:
                continue
            if dtype not in _NATIVE_DTYPES:
                continue
            initial = "0.0" if dtype == "float64" else "0"
            lines.append(f"    cdef {self.ctype(dtype)} {name} = {initial}")
        for statement in body:
            lines.extend(textwrap.indent(ast.unparse(statement), "    ").splitlines())
        lines.append("")

        sig = self._function_signature(cell, "calc") + ":"
        policy_i = self._policy_index_name(cell)
        zero = self._zero(cell.dtype)
        args = ", ".join(cell.parameters)
        suffix = "" if not args else ", " + args
        lines += [
            sig,
            f"    cdef {ct} val = <{ct}>{zero}",
            "    global _fallback_error, _fallback_exc, _fallback_tb",
            "    if _fallback_error != 0:",
            "        return val",
            "    with gil:",
            "        try:",
            (
                f"            val = {_category_helper_name('encode', original.result_category_domain)}("
                f"{pycalc}({policy_i}{suffix}))"
                if core_cache_python and original.result_category_domain is not None
                else f"            val = {pycalc}({policy_i}{suffix})"
            ),
            "        except BaseException as exc:",
            "            _fallback_exc = exc",
            "            _fallback_tb = exc.__traceback__",
            "            _fallback_error = 1",
            "    return val",
            "",
        ]
        return lines

    def _emit_python_key_formula(self, cell: NativeFormulaCell) -> list[str]:
        """Emit one GIL-held detached helper body for Python-key ``r_<uid>()``."""
        original = self.python_key_cells[cell.uid]
        fn = original.function
        if fn is None:
            raise MixedRecursiveNativeError(
                f"{cell.uid}: Python-key helper formula is missing source"
            )
        policy_index_name = self._policy_index_name(original)
        transformed = copy.deepcopy(fn)
        transformed = _PythonKeyMemoCallRewriter(
            self.python_key_cells, policy_index_name
        ).visit(transformed)
        transformed = _ObjectMemoCallRewriter(
            self.object_cells, policy_index_name
        ).visit(transformed)
        transformed = _ModelxLeafRewriter(
            self, original, policy_index_name
        ).visit(transformed)
        ref_vars = {
            spec.variable_name
            for spec in self.mixed_program.python_reference_specs
            if spec.owner_uid == cell.uid
        }
        if ref_vars:
            transformed = _PolicyIndexedPythonReferenceRewriter(
                ref_vars, policy_index_name
            ).visit(transformed)
        assert isinstance(transformed, ast.FunctionDef)
        ast.fix_missing_locations(transformed)
        body = list(transformed.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        body = [
            statement
            for statement in body
            if not (
                isinstance(statement, ast.Import)
                and any(alias.name == "cython" for alias in statement.names)
            )
            and not (
                isinstance(statement, ast.ImportFrom)
                and statement.module == "cython"
            )
        ]
        if not body:
            raise MixedRecursiveNativeError(
                f"{cell.uid}: empty Python-key helper formula body"
            )

        args_decl = [f"Py_ssize_t {policy_index_name}"]
        args_decl.extend(f"object {name}" for name in original.parameters)
        lines = [
            "@cython.boundscheck(True)",
            "@cython.wraparound(True)",
            f"cdef object pycalc_{cell.uid}({', '.join(args_decl)}):",
        ]
        for statement in body:
            lines.extend(textwrap.indent(ast.unparse(statement), "    ").splitlines())
        lines.append("")
        return lines


    def _emit_object_backed_formula(self, cell: NativeFormulaCell) -> list[str]:
        fn = cell.function
        if fn is None:
            raise MixedRecursiveNativeError(f"{cell.uid}: object-backed formula is missing source")
        policy_index_name = self._policy_index_name(cell)
        transformed = copy.deepcopy(fn)
        transformed = _PythonKeyMemoCallRewriter(
            self.python_key_cells, policy_index_name
        ).visit(transformed)
        transformed = _ObjectMemoCallRewriter(
            self.object_cells, policy_index_name
        ).visit(transformed)
        transformed = _ModelxLeafRewriter(self, cell, policy_index_name).visit(transformed)
        ref_vars = {
            spec.variable_name
            for spec in self.mixed_program.python_reference_specs
            if spec.owner_uid == cell.uid
        }
        if ref_vars:
            transformed = _PolicyIndexedPythonReferenceRewriter(
                ref_vars, policy_index_name
            ).visit(transformed)
        assert isinstance(transformed, ast.FunctionDef)
        ast.fix_missing_locations(transformed)
        body = list(transformed.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        body = [
            statement
            for statement in body
            if not (
                isinstance(statement, ast.Import)
                and any(alias.name == "cython" for alias in statement.names)
            )
            and not (
                isinstance(statement, ast.ImportFrom)
                and statement.module == "cython"
            )
        ]
        if not body:
            raise MixedRecursiveNativeError(f"{cell.uid}: empty object-backed formula body")
        # Object/ranked PyObject-backed P formulas still cross the same native
        # category ABI as scalar detached-P formulas.  Keep integer codes in the
        # compiler cache, but restore the original Python label before executing
        # the preserved Python body.  Otherwise a body such as ``j == 'LAST'``
        # silently observes ``1``/``2`` and can fall off the end returning None.
        core_cache_python = cell.uid in self.mixed_program.core_cache_cython_p_uids
        parameter_domains = tuple(cell.parameter_category_domains)
        if len(parameter_domains) < len(cell.parameters):
            parameter_domains = parameter_domains + (None,) * (
                len(cell.parameters) - len(parameter_domains)
            )
        args_decl = [f"Py_ssize_t {policy_index_name}"]
        category_argument_prelude: list[str] = []
        for name, dtype, did in zip(
            cell.parameters, cell.parameter_dtypes, parameter_domains
        ):
            if core_cache_python and did is not None:
                code_name = f"__mx_catcode_{name}"
                args_decl.append(f"{self.ctype(dtype)} {code_name}")
                category_argument_prelude.append(
                    f"    {name} = {_category_helper_name('decode', did)}({code_name})"
                )
            else:
                args_decl.append(f"{self.ctype(dtype)} {name}")
        lines = [
            "@cython.boundscheck(True)",
            "@cython.wraparound(True)",
            f"cdef object objcalc_{cell.uid}({', '.join(args_decl)}):",
            *category_argument_prelude,
        ]
        for statement in body:
            lines.extend(textwrap.indent(ast.unparse(statement), "    ").splitlines())
        lines.append("")
        return lines

    def _emit_ranked_object_backed_formula(self, cell: NativeFormulaCell) -> list[str]:
        """Emit one object/ranked-array miss body for the rank-1 GIL path."""
        fn = cell.function
        if fn is None:
            raise MixedRecursiveNativeError(f"{cell.uid}: ranked object formula is missing source")
        policy_index_name = self._policy_index_name(cell)
        transformed = _PythonKeyMemoCallRewriter(
            self.python_key_cells, policy_index_name
        ).visit(copy.deepcopy(fn))
        transformed = _ObjectMemoCallRewriter(
            self.object_cells, policy_index_name
        ).visit(transformed)
        transformed = _RankedNativeMemoCallRewriter(self.v).visit(transformed)
        ref_vars = {
            spec.variable_name
            for spec in self.mixed_program.python_reference_specs
            if spec.owner_uid == cell.uid
        }
        if ref_vars:
            transformed = _PolicyIndexedPythonReferenceRewriter(
                ref_vars, policy_index_name
            ).visit(transformed)
        assert isinstance(transformed, ast.FunctionDef)
        ast.fix_missing_locations(transformed)
        body = list(transformed.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if not body:
            raise MixedRecursiveNativeError(f"{cell.uid}: empty ranked object formula body")
        # Object/ranked PyObject-backed P formulas still cross the same native
        # category ABI as scalar detached-P formulas.  Keep integer codes in the
        # compiler cache, but restore the original Python label before executing
        # the preserved Python body.  Otherwise a body such as ``j == 'LAST'``
        # silently observes ``1``/``2`` and can fall off the end returning None.
        core_cache_python = cell.uid in self.mixed_program.core_cache_cython_p_uids
        parameter_domains = tuple(cell.parameter_category_domains)
        if len(parameter_domains) < len(cell.parameters):
            parameter_domains = parameter_domains + (None,) * (
                len(cell.parameters) - len(parameter_domains)
            )
        args_decl = [f"Py_ssize_t {policy_index_name}"]
        category_argument_prelude: list[str] = []
        for name, dtype, did in zip(
            cell.parameters, cell.parameter_dtypes, parameter_domains
        ):
            if core_cache_python and did is not None:
                code_name = f"__mx_catcode_{name}"
                args_decl.append(f"{self.ctype(dtype)} {code_name}")
                category_argument_prelude.append(
                    f"    {name} = {_category_helper_name('decode', did)}({code_name})"
                )
            else:
                args_decl.append(f"{self.ctype(dtype)} {name}")
        lines = [
            "@cython.boundscheck(True)",
            "@cython.wraparound(True)",
            f"cdef object objcalc_{cell.uid}({', '.join(args_decl)}):",
            *category_argument_prelude,
        ]
        for statement in body:
            lines.extend(textwrap.indent(ast.unparse(statement), "    ").splitlines())
        lines.append("")
        return lines

    @staticmethod
    def _insert_after_function_global(source: str, signature: str, line: str) -> str:
        start = source.find(signature)
        if start < 0:
            raise MixedRecursiveNativeError(f"cannot locate generated function {signature.strip()}")
        body_start = start + len(signature)
        if not source.startswith("    global ", body_start):
            raise MixedRecursiveNativeError(
                f"generated function {signature.strip()} has no leading global declaration"
            )
        end = source.find("\n", body_start)
        return source[: end + 1] + line + source[end + 1 :]

    def _integrate_object_memos(self, source: str) -> str:
        if not self.object_cells:
            return source
        emitter = self.object_memo_emitter
        first_newline = source.find("\n")
        source = source[: first_newline + 1] + emitter.import_source() + source[first_newline + 1 :]

        epoch_anchor = "cdef unsigned int _epoch = 1\n"
        if epoch_anchor not in source:
            raise MixedRecursiveNativeError("cannot locate epoch declaration for object memos")
        source = source.replace(
            epoch_anchor,
            emitter.helper_source() + epoch_anchor + emitter.declaration_source(),
            1,
        )

        init_anchor = "cpdef init_inputs(dict d):\n"
        object_runtime: list[str] = []
        for uid in self.mixed_program.object_backed_uids:
            object_runtime.extend(self._emit_object_backed_formula(self.object_cells[uid]))
        object_runtime_source = "\n".join(object_runtime)
        object_runtime_source += emitter.wrapper_source()
        object_runtime_source += emitter.clear_source()
        object_runtime_source += "\n" + emitter.stats_source()
        object_runtime_source += "\n" + emitter.reset_stats_source()
        object_runtime_source += (
            "\ncpdef clear_object_memo_cache():\n"
            "    _clear_object_memos()\n"
            "    return None\n\n"
        )
        if init_anchor not in source:
            raise MixedRecursiveNativeError("cannot locate init_inputs for object memo integration")
        source = source.replace(init_anchor, object_runtime_source + init_anchor, 1)
        source = self._insert_after_function_global(
            source, init_anchor, "    _clear_object_memos()\n"
        )

        next_epoch = "cdef inline void _next_epoch() noexcept nogil:\n"
        start = source.find(next_epoch)
        if start < 0:
            raise MixedRecursiveNativeError("cannot locate epoch advance for object memo integration")
        reset_anchor = "        _epoch = 1\n"
        reset = source.find(reset_anchor, start)
        if reset < 0:
            raise MixedRecursiveNativeError("cannot locate epoch wrap reset for object memos")
        generation_clear = "\n".join(emitter.generation_clear_lines(indent="        "))
        if generation_clear:
            source = source[:reset] + generation_clear + "\n" + source[reset:]
        return source

    def _integrate_python_key_memos(self, source: str) -> str:
        """Attach detached Python-key helper Cells to the scalar recursive module."""
        if not self.python_key_cells:
            return source
        emitter = self.python_key_memo_emitter

        epoch_anchor = "cdef unsigned int _epoch = 1\n"
        if epoch_anchor not in source:
            raise MixedRecursiveNativeError(
                "cannot locate epoch declaration for Python-key memos"
            )
        source = source.replace(
            epoch_anchor,
            epoch_anchor + emitter.declaration_source(),
            1,
        )

        runtime_lines: list[str] = []
        for uid in self.mixed_program.python_key_backed_uids:
            runtime_lines.extend(self._emit_python_key_formula(self.python_key_cells[uid]))
        runtime_source = "\n".join(runtime_lines)
        runtime_source += emitter.wrapper_source()
        runtime_source += emitter.clear_source()
        runtime_source += "\n" + emitter.stats_source()
        runtime_source += "\n" + emitter.reset_stats_source()
        runtime_source += (
            "\ncpdef clear_python_key_memo_cache():\n"
            "    _clear_python_key_memos()\n"
            "    return None\n\n"
        )

        init_anchor = "cpdef init_inputs(dict d):\n"
        if init_anchor not in source:
            raise MixedRecursiveNativeError(
                "cannot locate init_inputs for Python-key memo integration"
            )
        source = source.replace(init_anchor, runtime_source + init_anchor, 1)
        source = self._insert_after_function_global(
            source, init_anchor, "    _clear_python_key_memos()\n"
        )

        # _next_epoch runs nogil. Acquiring the GIL only on the uint32 wrap is
        # effectively free in ordinary runs and prevents an ancient epoch-1
        # Python dict from being mistaken for the new epoch-1 cache.
        next_epoch = "cdef inline void _next_epoch() noexcept nogil:\n"
        start = source.find(next_epoch)
        if start < 0:
            raise MixedRecursiveNativeError(
                "cannot locate epoch advance for Python-key memo integration"
            )
        reset_anchor = "        _epoch = 1\n"
        reset = source.find(reset_anchor, start)
        if reset < 0:
            raise MixedRecursiveNativeError(
                "cannot locate epoch wrap reset for Python-key memos"
            )
        wrap_clear = "        with gil:\n            _clear_python_key_memos()\n"
        source = source[:reset] + wrap_clear + source[reset:]
        return source


    def _integrate_ranked_python_memos(self, source: str) -> str:
        """Attach detached P formulas to the rank-1 NumPy execution path."""
        emitter = self.object_memo_emitter
        import_anchor = "import numpy as np\n"
        if import_anchor not in source:
            raise MixedRecursiveNativeError("cannot locate ranked NumPy import")

        declarations: list[str] = []
        if self.object_cells:
            declarations += [
                emitter.import_source().rstrip(),
                "cdef unsigned int _epoch = 1",
                "cdef long long _memo_error = 0",
                emitter.helper_source().rstrip(),
                emitter.declaration_source().rstrip(),
            ]
        declarations.append("cdef object _fallback_runtime = None")
        declarations.extend(
            f"cdef object {spec.variable_name} = None"
            for spec in self.mixed_program.python_reference_specs
        )
        category_helpers = self._category_bridge_helpers().rstrip()
        if category_helpers:
            declarations.append(category_helpers)
        declaration_source = "\n".join(x for x in declarations if x) + "\n"
        source = source.replace(import_anchor, import_anchor + declaration_source, 1)

        init_anchor = "cpdef init_inputs(dict d):\n"
        if init_anchor not in source:
            raise MixedRecursiveNativeError("cannot locate ranked init_inputs")
        runtime_lines: list[str] = []
        if self.object_cells:
            for uid in self.mixed_program.object_backed_uids:
                runtime_lines.extend(
                    self._emit_ranked_object_backed_formula(self.object_cells[uid])
                )
            runtime_lines.extend(emitter.wrapper_source().splitlines())
            runtime_lines.extend(emitter.clear_source().splitlines())
            runtime_lines.extend(emitter.stats_source().splitlines())
            runtime_lines.extend(emitter.reset_stats_source().splitlines())
            runtime_lines += [
                "cpdef clear_object_memo_cache():",
                "    _clear_object_memos()",
                "    return None",
                "",
            ]

        ref_specs = tuple(self.mixed_program.python_reference_specs)
        global_names = ["_fallback_runtime", *(spec.variable_name for spec in ref_specs)]
        zero_ranked: list[tuple[str, str]] = []
        for uid in sorted(self.v):
            cell = self.v[uid]
            if cell.arity == 0:
                zero_ranked.append((f"_ranked_has_{uid}", f"_ranked_value_{uid}"))
                global_names.extend(zero_ranked[-1])
        runtime_lines += [
            "cpdef bind_fallback_runtime(object runtime):",
            "    global " + ", ".join(global_names),
        ]
        if self.object_cells:
            runtime_lines.append("    _clear_object_memos()")
        for has_name, value_name in zero_ranked:
            runtime_lines += [f"    {has_name} = False", f"    {value_name} = None"]
        for uid in sorted(self.v):
            if self.v[uid].arity:
                runtime_lines.append(f"    _ranked_cache_{uid}.clear()")
        runtime_lines.append("    _fallback_runtime = runtime")
        if ref_specs:
            runtime_lines += [
                "    if runtime is None:",
                "        raise RuntimeError('detached ranked P references require bind_fallback_runtime(runtime)')",
            ]
            for spec in ref_specs:
                runtime_lines.append(
                    f"    {spec.variable_name} = runtime.resolve_python_reference_series("
                    f"{spec.owner_uid!r}, {spec.root_name!r}, {spec.attribute_path!r}, "
                    f"{bool(spec.invoke_zero_arg)!r}, {spec.invoke_args!r}, {spec.invoke_kwargs!r})"
                )
        runtime_lines += [
            "    return None",
            "",
            "cpdef dict fallback_stats():",
            "    if _fallback_runtime is None:",
            "        return {}",
            "    return _fallback_runtime.stats()",
            "",
        ]
        source = source.replace(
            init_anchor, "\n".join(runtime_lines) + init_anchor, 1
        )

        init_prefix: list[str] = ["    global _fallback_runtime"]
        if self.object_cells:
            init_prefix += [
                "    global _epoch, _memo_error",
                "    _clear_object_memos()",
                "    _epoch = 1",
                "    _memo_error = 0",
            ]
        for spec in ref_specs:
            init_prefix += [
                f"    global {spec.variable_name}",
                f"    {spec.variable_name} = None",
            ]
        init_prefix.append("    _fallback_runtime = None")
        source = source.replace(
            init_anchor, init_anchor + "\n".join(init_prefix) + "\n", 1
        )

        if self.object_cells:
            out = self.v[self.formula.output_uid]
            npdt = (
                "np.float64" if out.dtype == "float64"
                else "np.bool_" if out.dtype == "bool"
                else "np.int64"
            )
            result_line = (
                f"    cdef cnp.ndarray result = np.asarray(r_{out.uid}(), dtype={npdt})\n"
            )
            if result_line not in source:
                raise MixedRecursiveNativeError("cannot locate ranked run_all result")

            if self.mixed_program.python_output_uid is not None:
                py_uid = self.mixed_program.python_output_uid
                expected = self.mixed_program.python_output_shape
                if expected is None:
                    raise MixedRecursiveNativeError(
                        "Python-backed selected output requires fixed geometry"
                    )
                selected = self.object_cells.get(py_uid)
                if selected is None:
                    raise MixedRecursiveNativeError(
                        f"Python-backed selected output {py_uid!r} lacks object memo"
                    )
                cast = (
                    ", dtype=np.float64" if selected.dtype == "float64"
                    else ", dtype=np.bool_" if selected.dtype == "bool"
                    else ", dtype=np.int64" if selected.dtype == "int64"
                    else ""
                )
                source = source.replace(
                    result_line,
                    "    cdef object raw\n"
                    "    cdef cnp.ndarray result\n"
                    "    global _memo_error\n"
                    "    _memo_error = 0\n"
                    f"    raw = r_{py_uid}(0)\n"
                    "    if _memo_error != 0:\n"
                    "        raise RuntimeError(f'object memo failure code {_memo_error}')\n"
                    f"    result = np.asarray(raw{cast})\n",
                    1,
                )
                old_shape = (
                    f"    if result.ndim != 1 or result.shape[0] != {int(out.logical_result_shape[0])}:\n"
                    "        raise ValueError('rank-1 output shape mismatch')\n"
                )
                shape_checks = [f"result.ndim != {len(expected)}"] + [
                    f"result.shape[{i}] != {int(dim)}"
                    for i, dim in enumerate(expected)
                ]
                new_shape = (
                    f"    if {' or '.join(shape_checks)}:\n"
                    "        raise ValueError('Python-backed output shape mismatch')\n"
                )
                if old_shape not in source:
                    raise MixedRecursiveNativeError(
                        "cannot locate ranked driver output shape check"
                    )
                source = source.replace(old_shape, new_shape, 1)
            else:
                source = source.replace(
                    result_line,
                    "    cdef object raw\n"
                    "    cdef cnp.ndarray result\n"
                    "    global _memo_error\n"
                    "    _memo_error = 0\n"
                    f"    raw = r_{out.uid}()\n"
                    "    if _memo_error != 0:\n"
                    "        raise RuntimeError(f'object memo failure code {_memo_error}')\n"
                    f"    result = np.asarray(raw, dtype={npdt})\n",
                    1,
                )
        return source

    def _rewrite_scalar_python_output_runner(self, source: str) -> str:
        """Return one selected Python-object value per policy.

        Python-object outputs cannot use the ranked driver because their native
        dependencies still require the policy index for point-scoped inputs.
        Keep the scalar recursive engine, advance its memo epoch once per policy,
        and call the detached object-memo wrapper under the GIL.
        """
        py_uid = self.mixed_program.python_output_uid
        if py_uid is None:
            return source
        selected = self.object_cells.get(py_uid)
        if selected is None:
            raise MixedRecursiveNativeError(
                f"Python-backed selected output {py_uid!r} lacks object memo"
            )
        start = source.find("cpdef cnp.ndarray run_all():\n")
        if start < 0:
            raise MixedRecursiveNativeError("cannot locate scalar run_all for Python output")
        end = source.find("cpdef double bench(long long reps):\n", start)
        if end < 0:
            raise MixedRecursiveNativeError("cannot locate scalar bench after run_all")
        runner = (
            "cpdef object run_all():\n"
            "    cdef list result = []\n"
            "    cdef Py_ssize_t i\n"
            "    global _memo_error, _guard_error\n"
            "    _memo_error = 0\n"
            "    _guard_error = 0\n"
            "    for i in range(_n):\n"
            "        _next_epoch()\n"
            f"        result.append(r_{py_uid}(i))\n"
            "        if _memo_error != 0 or _guard_error != 0:\n"
            "            break\n"
            "    _fallback_raise_if_needed()\n"
            "    if _memo_error != 0:\n"
            "        raise RuntimeError(f'recursive memo failure code {_memo_error}')\n"
            "    if _guard_error != 0:\n"
            "        raise ValueError(f'native formula guard failed with code {_guard_error}')\n"
            "    return result\n\n"
        )
        return source[:start] + runner + source[end:]

    def _category_bridge_helpers(self) -> str:
        ids = tuple(self.mixed_program.category_bridge_domain_ids)
        if not ids:
            return ""
        domains = dict(self.mixed_program.formula_program.category_domains)
        lines: list[str] = [
            "# Dev136: reversible category ABI at Cython-under-GIL formula boundaries.",
        ]
        for did in ids:
            labels = domains.get(did)
            if labels is None:
                raise MixedRecursiveNativeError(f"missing category domain {did!r}")
            enc = _category_helper_name("encode", did)
            dec = _category_helper_name("decode", did)
            lines.append(f"cdef inline long long {enc}(object value) except? -1:")
            for code, label in enumerate(labels):
                lines.append(f"    if value == {label!r}:")
                lines.append(f"        return {code}")
            lines.append(
                f"    raise ValueError('category value outside proven domain {did}')"
            )
            lines.append("")
            lines.append(f"cdef inline object {dec}(long long code):")
            for code, label in enumerate(labels):
                lines.append(f"    if code == {code}:")
                lines.append(f"        return {label!r}")
            lines.append(
                f"    raise ValueError('category code outside proven domain {did}')"
            )
            lines.append("")
        return "\n".join(lines) + "\n"

    def _emit_modelx_fallback_formula(self, cell: NativeFormulaCell) -> list[str]:
        raise MixedRecursiveNativeError(
            f"{cell.uid}: live modelx formula execution is disabled; P must use detached Python semantics"
        )


    def _emit_formula(self, cell: NativeFormulaCell) -> list[str]:
        if cell.uid not in self.mixed_program.execution_class_plan.cells:
            return SourcePreservingDenseCythonGenerator._emit_formula(self, cell)
        cls = self.mixed_program.execution_class_plan.cell(cell.uid).execution_class
        if cls != P:
            return SourcePreservingDenseCythonGenerator._emit_formula(self, cell)
        if cell.uid in self.mixed_program.gil_backed_cells:
            return self._emit_gil_backed_formula(cell)
        raise MixedRecursiveNativeError(
            f"{cell.uid}: P Cell lacks detached Python-semantic source"
        )

    def source(self, module_name: str = "mixed_recursive_native") -> str:
        source = SourcePreservingDenseCythonGenerator.source(self, module_name)
        if (
            self.mixed_program.python_backed_uids
            or self.mixed_program.python_key_backed_uids
        ) and "cimport cython\n" not in source:
            head, separator, body = source.partition("\n")
            source = head + separator + "cimport cython\n" + body
        if self._ranked_program:
            if self.python_key_cells:
                raise MixedRecursiveNativeError(
                    "Part 2A Python-key helpers are not wired into the ranked driver yet"
                )
            if self.openmp:
                raise MixedRecursiveNativeError("OpenMP scalar driver does not support ranked outputs")
            if not self.mixed_program.python_backed_uids:
                return source
            return self._integrate_ranked_python_memos(source)
        source = self._integrate_object_memos(source)
        source = self._integrate_python_key_memos(source)
        if self.diagnostic_instrumentation:
            diag_anchor = "cdef Py_ssize_t _n = 0\n"
            if diag_anchor not in source:
                raise MixedRecursiveNativeError("cannot locate diagnostic runtime globals")
            source = source.replace(
                diag_anchor, diag_anchor + self._diagnostic_declarations(), 1
            )
            diag_init_anchor = "cpdef init_inputs(dict d):\n"
            if diag_init_anchor not in source:
                raise MixedRecursiveNativeError("cannot locate diagnostic init_inputs")
            source = source.replace(
                diag_init_anchor, self._diagnostic_helpers() + diag_init_anchor, 1
            )
        category_helpers = self._category_bridge_helpers()
        if category_helpers:
            helper_anchor = "cdef unsigned int _epoch = 1\n"
            if helper_anchor not in source:
                raise MixedRecursiveNativeError("cannot locate module declarations for category bridge helpers")
            source = source.replace(helper_anchor, category_helpers + helper_anchor, 1)

        # A pure N0 selection stays exactly on the pre-existing native path: no
        # fallback globals, no GIL acquisition and no Python runtime binding.
        if not self.mixed_program.python_backed_uids:
            if self.openmp:
                from .openmp_native import enable_dense_openmp
                return enable_dense_openmp(source)
            return source
        anchor = "cdef Py_ssize_t _n = 0\n"
        if anchor not in source:
            raise MixedRecursiveNativeError("cannot locate generated recursive runtime globals")
        ref_specs = tuple(self.mixed_program.python_reference_specs)
        ref_globals = "".join(
            f"cdef object {spec.variable_name} = None\n" for spec in ref_specs
        )
        globals_src = (
            anchor
            + "cdef object _fallback_runtime = None\n"
            + "cdef object _fallback_exc = None\n"
            + "cdef object _fallback_tb = None\n"
            + "cdef long long _fallback_error = 0\n"
            + ref_globals
        )
        source = source.replace(anchor, globals_src, 1)

        init_anchor = "cpdef init_inputs(dict d):\n"
        global_names = [
            "_fallback_runtime", "_fallback_exc", "_fallback_tb", "_fallback_error",
            *(spec.variable_name for spec in ref_specs),
        ]
        helper_lines = [
            "cpdef bind_fallback_runtime(object runtime):",
            "    global " + ", ".join(global_names),
            *(["    _clear_object_memos()"] if self.object_cells else []),
            *(["    _clear_python_key_memos()"] if self.python_key_cells else []),
            "    _fallback_runtime = runtime",
            "    _fallback_exc = None",
            "    _fallback_tb = None",
            "    _fallback_error = 0",
        ]
        if ref_specs:
            helper_lines += [
                "    if runtime is None:",
                "        raise RuntimeError('detached Python P references require bind_fallback_runtime(runtime)')",
            ]
            for spec in ref_specs:
                helper_lines.append(
                    f"    {spec.variable_name} = runtime.resolve_python_reference_series("
                    f"{spec.owner_uid!r}, {spec.root_name!r}, {spec.attribute_path!r}, "
                    f"{bool(spec.invoke_zero_arg)!r}, {spec.invoke_args!r}, {spec.invoke_kwargs!r})"
                )
        helper_lines += [
            "    return None",
            "",
            "cdef void _fallback_raise_if_needed():",
            "    global _fallback_error, _fallback_exc, _fallback_tb",
            "    cdef object exc",
            "    cdef object tb",
            "    if _fallback_error == 0:",
            "        return",
            "    exc = _fallback_exc",
            "    tb = _fallback_tb",
            "    _fallback_error = 0",
            "    _fallback_exc = None",
            "    _fallback_tb = None",
            "    if exc is None:",
            "        raise RuntimeError('Python-backed Cell failed without captured exception')",
            "    raise exc.with_traceback(tb)",
            "",
            "cpdef dict fallback_stats():",
            "    if _fallback_runtime is None:",
            "        return {}",
            "    return _fallback_runtime.stats()",
            "",
        ]
        helpers = "\n".join(helper_lines) + "\n"
        if init_anchor not in source:
            raise MixedRecursiveNativeError("cannot locate generated init_inputs")
        source = source.replace(init_anchor, helpers + init_anchor, 1)

        run_exit = (
            "    if _memo_error != 0:\n"
            "        raise RuntimeError(f'recursive memo failure code {_memo_error}')\n"
        )
        run_exit_repl = (
            "    _fallback_raise_if_needed()\n"
            "    if _memo_error != 0:\n"
            "        raise RuntimeError(f'recursive memo failure code {_memo_error}')\n"
        )
        if source.count(run_exit) < 2:
            raise MixedRecursiveNativeError("cannot locate run_all/bench error exits")
        source = source.replace(run_exit, run_exit_repl, 2)
        if self.mixed_program.python_output_uid is not None:
            source = self._rewrite_scalar_python_output_runner(source)
        return source

    def declaration_source(self) -> str:
        source = super().declaration_source()
        if self.diagnostic_instrumentation:
            source += "cpdef dict recursive_stats()\ncpdef reset_recursive_stats()\n"
        if self.object_cells:
            for uid in self.mixed_program.object_backed_uids:
                cell = self.object_cells[uid]
                args = ["Py_ssize_t policy_i"] + [
                    f"{self.ctype(dtype)} {name}"
                    for name, dtype in zip(cell.parameters, cell.parameter_dtypes)
                ]
                source += f"cdef object r_{uid}({', '.join(args)})\n"
                source += f"cdef object objcalc_{uid}({', '.join(args)})\n"
            source += (
                "cpdef dict object_memo_stats()\n"
                "cpdef reset_object_memo_stats()\n"
                "cpdef clear_object_memo_cache()\n"
            )
        if self.python_key_cells:
            for uid in self.mixed_program.python_key_backed_uids:
                cell = self.python_key_cells[uid]
                args = ["Py_ssize_t policy_i"] + [
                    f"object {name}" for name in cell.parameters
                ]
                source += f"cdef object r_{uid}({', '.join(args)})\n"
                source += f"cdef object pycalc_{uid}({', '.join(args)})\n"
            source += (
                "cpdef dict python_key_memo_stats()\n"
                "cpdef reset_python_key_memo_stats()\n"
                "cpdef clear_python_key_memo_cache()\n"
            )
        if not (
            self.mixed_program.python_backed_uids
            or self.mixed_program.python_key_backed_uids
        ):
            return source
        return source + "cpdef bind_fallback_runtime(object runtime)\ncpdef dict fallback_stats()\n"


__all__ = [
    "MixedRecursiveNativeError",
    "FallbackTargetResolutionError",
    "FallbackSpaceHop",
    "FallbackTargetSpec",
    "FallbackTargetBlocker",
    "FallbackTargetPlan",
    "build_fallback_target_plan",
    "FallbackPolicyRuntime",
    "MixedRecursiveProgram",
    "build_mixed_recursive_program_from_fast_plan",
    "MixedRecursiveCythonGenerator",
]
