"""Atomic prepared-input binding and reusable per-policy scratch.

Runtime loading uses only the manifest, NumPy and generated binary package.
It does not import modelx, modelx-cython or the legacy compiler.
"""
from pathlib import Path
import importlib
import json
import math
import numbers
import sys
import threading
import numpy as np
from .contracts import ContractError
from .input_codec import NativeInputCodec, _INDEXED_ARRAYS
from .lookups import (Curve, CurveTable, IndexedFloat1, IndexedFloat2,
                      IndexedInt1, IndexedInt2)

class CompiledKernel(NativeInputCodec):
    def __init__(self, artifact, records, *, python_bindings=None):
        self.artifact = Path(artifact).resolve()
        self.manifest = json.loads((self.artifact / 'manifest.json').read_text())
        if self.manifest.get('schema') != 'mxcf.scalar_kernel.v1':
            raise ContractError('Unsupported kernel artifact schema')
        sys.path.insert(0, str(self.artifact))
        package = self.manifest['package']
        self._lookups = importlib.import_module(package + '._lookups')
        self._module = importlib.import_module(package + '._kernel')
        self._lock = threading.RLock()
        self._open_scope = None
        self._records = self._validate(records)
        self._kernel = self._module.Kernel(self._records)
        self._python_scope = None
        if self.manifest.get('python_cells'):
            from .python_calculator import PythonScope
            if python_bindings is not None and len(python_bindings) != len(self._records):
                raise ContractError('Python binding count must match policy count')
            self._python_scope = PythonScope(self.artifact, self.manifest, self._kernel, python_bindings)
            self._kernel.set_python_scope(self._python_scope)

    def _require_root(self):
        if self._open_scope is not None:
            raise RuntimeError('Cannot reset, rebind or select a root inside an active evaluation scope')

    def _update_values(self, changes, policy=None):
        """Change numeric bindings only; retain identical packed table objects."""
        infos = self.manifest['inputs']
        if set(changes)-set(infos):
            raise ContractError('Unknown input names')
        if any(infos[k]['kind'] in ('curve','curve_table','indexed_array') for k in changes):
            raise ContractError('Use bind() to replace table data')
        with self._lock:
            self._require_root()
            if policy is not None and (isinstance(policy,bool) or not isinstance(policy,numbers.Integral) or not 0<=policy<len(self._records)):
                raise IndexError('Policy position outside the bound portfolio')
            rows = list(self._records)
            positions = range(len(rows)) if policy is None else (int(policy),)
            locations = {k:i for i,k in enumerate(infos)}
            for i in positions:
                row = list(rows[i])
                record = dict(zip(infos,row))
                for k,v in changes.items(): row[locations[k]] = self._scalar_value(k,infos[k],v,record)
                rows[i] = tuple(row)
            updated = tuple(rows)
            self._kernel.replace_records(updated)
            self._records = updated

    def update_inputs(self, **values):
        """Integration API: replace scalar/object bindings without repacking tables."""
        self._update_values(values)

    def set_parameters(self, **values):
        """Update declared model parameters without repacking or rereading tables."""
        if set(values)-set(self.manifest.get('parameters',())):
            raise ContractError('Only declared model parameters may be set here')
        self._update_values(values)

    def set_policy_inputs(self, policy, **values):
        """Update one policy's scalar inputs; model formulas perform any selection."""
        if set(values)&set(self.manifest.get('parameters',())):
            raise ContractError('Use set_parameters() for portfolio-wide parameters')
        self._update_values(values, policy)

    def bind(self, records, *, python_bindings=None):
        """Atomically replace the native and detached-Python input snapshot.

        A mixed model with external receiver references must supply both sides.
        This prevents silently retaining stale Data tables in its Python receiver.
        """
        converted = self._validate(records)
        with self._lock:
            self._require_root()
            candidate = None
            if self._python_scope is not None:
                from .python_calculator import PythonScope
                if self.manifest['python_references'] and python_bindings is None:
                    raise ContractError('Rebinding this mixed kernel requires matching Python receiver bindings')
                if python_bindings is not None and len(python_bindings) != len(converted):
                    raise ContractError('Python binding count must match policy count')
                candidate = PythonScope(self.artifact, self.manifest, self._kernel, python_bindings)
                candidate.calls.update(self._python_scope.calls)
            self._kernel.replace_records(converted)
            if candidate is not None:
                self._kernel.set_python_scope(candidate)
                self._python_scope = candidate
            self._records = converted

    def reset(self):
        with self._lock:
            self._require_root()
            self._kernel.reset()

    def run_all(self, table=None):
        """Census-compatible sum of every numeric entry in the chosen output table."""
        return self.project(table, sums=True)

    def project(self, table=None, *, sums=False):
        table = table or next(iter(self.manifest['outputs']))
        if table not in self.manifest['outputs']:
            raise ContractError('Unknown output table')
        with self._lock:
            self._require_root()
            return self._kernel.run(table, sums)

    def run_scalar(self, name):
        if name not in self.manifest['scalar_outputs']:
            raise ContractError('Unknown scalar output')
        with self._lock:
            self._require_root()
            return self._kernel.run_scalar(name)

    def _run_object_outputs(self, name, detach, consume=None):
        """Traverse an object Cell once per policy with in-scope detachment.

        This is an internal building block for the exported collect contract.
        ``detach`` is the modelx/export rule for a completed policy output;
        it must run before the working context advances to the next policy.
        """
        if (name not in self.manifest.get('cell_outputs', ())
                or name not in self.manifest.get('object_cells', ())
                or self.manifest['cells'][name]['params']
                or self.manifest['cells'][name]['result']['kind'] != 'object'):
            raise ContractError('Expected a zero-argument object Cell output')
        with self._lock:
            self._require_root()
            return self._kernel.run_object_outputs(name, detach, consume)

    def _compiled_collect_keys(self):
        """Return stable exact-int map domains, or use ordinary collect traversal.

        This inspects already bound references. It neither evaluates Cells nor
        changes their inputs, and fallback preserves arbitrary Python iterables.
        """
        plan = self.manifest.get('compiled_collect_map')
        if not plan or self._python_scope is None:
            return None
        result = {}
        for source in plan['sources']:
            if source in self.manifest['inputs']:
                position = list(self.manifest['inputs']).index(source)
                values = (record[position] for record in self._records)
            else:
                bindings = self._python_scope.bindings
                if len(bindings) != len(self._records):
                    return None
                values = (binding.get(source) for binding in bindings)
            first = next(values, None)
            if (type(first) is not tuple
                    or any(type(key) is not int or not -(1 << 63) <= key < (1 << 63)
                           for key in first)
                    or len(set(first)) != len(first)):
                return None
            for value in values:
                if type(value) is not tuple or (value is not first and value != first):
                    return None
            result[source] = first
        return result

    def _run_compiled_collect_map(self, keys):
        with self._lock:
            self._require_root()
            return self._kernel.run_compiled_collect_map(keys)

    def project_one(self, policy, table=None):
        table = table or next(iter(self.manifest['outputs']))
        if table not in self.manifest['outputs']:
            raise ContractError('Unknown output table')
        with self._lock:
            self._require_root()
            return self._kernel.run_one(int(policy), table)

    def run_scalar_one(self, policy, name):
        if name not in self.manifest['scalar_outputs']:
            raise ContractError('Unknown scalar output')
        with self._lock:
            self._require_root()
            return self._kernel.run_scalar_one(int(policy), name)

    def evaluate(self, cell, args=(), policy=0):
        """Diagnostic Cell evaluation; starts an independent memo epoch."""
        if cell not in self.manifest['cells']:
            raise ContractError('Cell is not in the compiled output closure')
        if self.manifest.get('object_cells'):
            with self.scope(policy) as scope:
                return scope.call(cell, *args)
        parameters = self.manifest['cells'][cell]['params']
        if len(args) > len(parameters):
            raise TypeError('Too many arguments')
        canonical = list(args)
        for p in parameters[len(canonical):]:
            if p['default'] is None:
                raise TypeError('Missing required argument ' + p['name'])
            canonical.append(float(p['default']) if p['type']['kind']=='float' else int(p['default']))
        for p, value in zip(parameters, canonical):
            if p['type']['kind'] == 'float':
                if not isinstance(value, numbers.Real) or not math.isfinite(value):
                    raise ContractError('Finite numeric argument required')
            elif not isinstance(value, numbers.Integral):
                raise ContractError('Integral argument required')
            elif not -(1 << 63) <= int(value) < (1 << 63):
                raise ContractError('Argument outside signed int64')
            if p['type']['kind'] == 'enum' and int(value) not in p['type']['members'].values():
                raise IndexError('Argument outside declared category domain')
            if p['type']['kind'] == 'bool' and value not in (True, False):
                raise ContractError('Boolean argument required')
        with self._lock:
            self._require_root()
            return self._kernel.evaluate(policy, cell, tuple(canonical))

    def scope(self, policy=0):
        """Internal shared evaluation scope; nested Cell calls reuse one epoch.

        This is a runtime boundary, not a new model-author input API. The generic
        modelx-shaped facade will own scopes in M3. Switching active ItemSpaces
        is deliberately rejected instead of overwriting live scratch.
        """
        if self._python_scope is None:
            raise ContractError('Shared scope requires a mixed artifact')
        return _EvaluationScope(self, policy)

    def fork_item(self, policy=0, *, input_overrides=None, python_overrides=None,
                  _native_kernel=None, **parameters):
        """Own one ItemSpace cache, sharing immutable tables with the parent.

        No repacking, formula reconstruction or second Python result cache. Each
        logical ItemSpace receives its own compiled context, as modelx does.
        """
        with self._lock:
            self._require_root()
            if isinstance(policy, bool) or not isinstance(policy, numbers.Integral) or not 0 <= policy < len(self._records):
                raise IndexError('Policy position outside the bound portfolio')
            unbounded_point = (self.manifest.get('model', {}).get('point_parameter')
                               if self.manifest.get('model', {}).get('point_table') is None
                               and not self.manifest.get('model', {}).get('static_projection')
                               else None)
            allowed = set(self.manifest.get('parameters', ()))
            if unbounded_point is not None:
                allowed.add(unbounded_point)
            if set(parameters) - allowed:
                raise ContractError('Unknown ItemSpace parameter')
            overrides = dict(input_overrides or {})
            python_overrides = dict(python_overrides or {})
            if set(python_overrides) - set(self.manifest.get('python_references', ())):
                raise ContractError('ItemSpace Python overrides must be declared references')
            if _native_kernel is not None and (parameters or overrides or
                    python_overrides or type(_native_kernel) is not self._module.Kernel):
                raise ContractError('Existing native ItemSpace owner cannot change its inputs')
            if not parameters and not overrides:
                record_tuple = self._records[policy]
            else:
                infos = self.manifest['inputs']
                record = dict(zip(infos, self._records[policy]))
                for name, value in parameters.items():
                    record[name] = self._scalar_value(name, infos[name], value, record)
                eligible = {name for name, binding in
                            self.manifest.get('model', {}).get('bindings', {}).items()
                            if binding['kind'] == 'row_reference'}
                eligible.update(name for name in self.manifest.get('model', {}).get(
                    'required_object_bindings', ()) if name in infos
                    and infos[name]['kind'] == 'object')
                if set(overrides) - eligible:
                    raise ContractError('ItemSpace input overrides must be row or object references')
                for name, value in overrides.items():
                    record[name] = self._scalar_value(name, infos[name], value, record)
                # A dynamic Space formula can replace a row's object references.
                # Derived numeric views follow that replacement in this owner.
                for name, binding in self.manifest.get('model', {}).get(
                        'native_row_references', {}).items():
                    column = binding['column']
                    if column not in overrides:
                        continue
                    value = overrides[column]
                    field = binding['field']
                    if field is not None:
                        try:
                            value = value[field]
                        except (KeyError, IndexError, TypeError) as exc:
                            raise ContractError(name + ': missing numeric row field') from exc
                    info = infos[name]
                    if info['kind'] == 'indexed_array':
                        record[name] = getattr(self._lookups, info['ctype'])(value)
                    else:
                        record[name] = self._scalar_value(name, info, value, record)
                record_tuple = tuple(record[n] for n in infos)
            child = object.__new__(type(self))
            for name in ('artifact', 'manifest', '_lookups', '_module'):
                setattr(child, name, getattr(self, name))
            child._lock, child._open_scope = self._lock, None
            child._records = (record_tuple,)
            child._kernel = (_native_kernel if _native_kernel is not None
                             else child._module.Kernel(child._records))
            child._python_scope = None
            child._lazy_python_source = None
            if self._python_scope is not None:
                bindings = [self._python_scope.bindings[policy]] if self._python_scope.bindings else None
                if (bindings is not None and unbounded_point in parameters
                        and unbounded_point not in self.manifest['inputs']):
                    bindings = [{**bindings[0], unbounded_point: parameters[unbounded_point]}]
                if python_overrides:
                    bindings = [{**(bindings[0] if bindings else {}), **python_overrides}]
                child._lazy_python_source = (bindings, self._python_scope.program)
            if _native_kernel is None:
                return ItemContext(child)
            return ItemContext(child, selected_native=True)

    def stats(self):
        with self._lock:
            result = self._kernel.stats()
            if self.manifest.get('object_cells'):
                result['object_cells'] = self._kernel.object_stats()
            if self._python_scope is not None:
                result['python_body_calls'] = dict(self._python_scope.calls)
            return result


def save_inputs(records, destination):
    """Portable prepared inputs: JSON scalars/labels + NPZ numeric arrays; no arbitrary-object pickle."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    curves, arrays, rows = {}, {}, []
    for r in records:
        row = {}
        for name, value in r.items():
            if type(value) in (Curve, CurveTable, *_INDEXED_ARRAYS.values()):
                if id(value) not in curves:
                    i = len(curves)
                    curves[id(value)] = i
                    if type(value) is Curve:
                        arrays[f'x{i}'], arrays[f'y{i}'] = value.arrays
                    elif type(value) is CurveTable:
                        arrays[f'g{i}'], arrays[f'b{i}'], arrays[f'x{i}'], arrays[f'y{i}'] = value.arrays
                    else:
                        arrays[f'a{i}'] = value.arrays[0]
                if type(value) is Curve:
                    row[name] = {'curve': curves[id(value)]}
                elif type(value) is CurveTable:
                    row[name] = {'curve_table': curves[id(value)], 'labels': value.labels}
                else:
                    row[name] = {'indexed_array': curves[id(value)], 'type': type(value).__name__}
            elif isinstance(value, str):
                row[name] = value
            elif isinstance(value, (bool, np.bool_)):
                row[name] = bool(value)
            elif isinstance(value, numbers.Integral):
                row[name] = int(value)
            else:
                row[name] = float(value)
        rows.append(row)
    np.savez(destination / 'arrays.npz', **arrays)
    (destination / 'inputs.json').write_text(json.dumps({'schema': 'mxcf.prepared.v1', 'records': rows}, allow_nan=False))


def load_inputs(source):
    """Load the portable bundle without pandas, modelx, or product modules."""
    source = Path(source)
    meta = json.loads((source / 'inputs.json').read_text())
    if meta.get('schema') != 'mxcf.prepared.v1':
        raise ContractError('Unsupported prepared input schema')
    curves = {}
    result = []
    with np.load(source / 'arrays.npz', allow_pickle=False) as arrays:
        for r in meta['records']:
            row = {}
            for name, value in r.items():
                if isinstance(value, dict):
                    kind = ('indexed_array' if 'indexed_array' in value else
                            'curve' if 'curve' in value else 'curve_table')
                    i = value[kind]
                    if i not in curves:
                        if kind == 'indexed_array':
                            cls = _INDEXED_ARRAYS.get(value.get('type'))
                            if cls is None:
                                raise ContractError('Unknown prepared numeric array type')
                            curves[i] = cls(arrays[f'a{i}'])
                        else:
                            curves[i] = Curve(arrays[f'x{i}'], arrays[f'y{i}']) if kind == 'curve' else CurveTable(arrays[f'g{i}'], arrays[f'b{i}'], arrays[f'x{i}'], arrays[f'y{i}'], value.get('labels'))
                    value = curves[i]
                row[name] = value
            result.append(row)
    return result


class _EvaluationScope:
    def __init__(self, owner, policy):
        self.owner, self.policy, self.active = owner, policy, False

    def __enter__(self):
        self.owner._lock.acquire()
        try:
            self.owner._require_root()
            self.owner._kernel.select_scope(self.policy)
            self.owner._open_scope = self
            self.active = True
            return self
        except BaseException:
            self.owner._lock.release()
            raise

    def __exit__(self, *exc):
        self.owner._open_scope = None
        self.active = False
        self.owner._lock.release()

    def call(self, cell, *args, **kwargs):
        if not self.active:
            raise RuntimeError('Evaluation scope is not active')
        if cell not in self.owner.manifest['cells']:
            raise ContractError('Cell is not in the compiled closure')
        # The proxy binds defaults/keywords and encodes/decodes enum/key values.
        return self.owner._python_scope.entry(cell)(*args, **kwargs)

    def __getattr__(self, name):
        if name in self.owner.manifest['cells']:
            return lambda *args, **kwargs: self.call(name, *args, **kwargs)
        raise AttributeError(name)


class ItemContext:
    """Persistent compiled cache for one ItemSpace/revision. Internal faÃ§ade API."""
    def __init__(self, owner, *, selected_native=False):
        self._owner = owner
        # Native scalar roots need neither signature proxies nor Enum resolution.
        # Construct the Python boundary only if a later public entry uses it.
        self._codec = owner._python_scope
        self._selected = selected_native
        self._selected_native = selected_native

    def _ensure_codec(self):
        if self._codec is None:
            from .python_calculator import CellCodec
            owner = self._owner
            self._codec = CellCodec(owner.manifest, owner._kernel, owner._records[0])
        return self._codec

    def _ensure_python_scope(self):
        owner = self._owner
        if owner._python_scope is not None or owner._lazy_python_source is None:
            return
        from .python_calculator import PythonScope
        bindings, program = owner._lazy_python_source
        scope = PythonScope(owner.artifact, owner.manifest, owner._kernel, bindings,
                            program=program)
        owner._kernel.set_python_scope(scope)
        owner._python_scope = scope
        self._codec = scope

    def _select_for(self, name):
        owner = self._owner
        native = (name in owner.manifest.get('native_scalar_outputs', ())
                  and hasattr(owner._kernel, 'select_native_scope'))
        if not self._selected:
            if native:
                owner._kernel.select_native_scope(0)
                self._selected_native = True
            else:
                self._ensure_python_scope()
                owner._kernel.select_scope(0)
            self._selected = True
        elif self._selected_native and not native:
            self._ensure_python_scope()
            owner._kernel.attach_python_scope_current()
            self._selected_native = False

    def call(self, name, *args, **kwargs):
        owner = self._owner
        with owner._lock:
            self._select_for(name)
            if name in owner.manifest['outputs']:
                if args or kwargs: raise TypeError('Output tables have no arguments')
                return owner._kernel.call_table(name)
            if name in owner.manifest['scalar_outputs']:
                if args or kwargs: raise TypeError('Scalar output has no arguments')
                value = owner._kernel.call_scalar(name)
                result = owner.manifest['cells'][name]['result']
                if result['kind'] == 'bool':
                    return bool(value)
                if result['kind'] not in ('enum', 'table_key'):
                    return value
                return self._ensure_codec().decode(result, value)
            if name not in owner.manifest['cells']:
                raise AttributeError(name)
            return self._ensure_codec().entry(name)(*args, **kwargs)

    def reset(self):
        owner = self._owner
        with owner._lock:
            owner._require_root()
            selected, native = self._selected, self._selected_native
            self._selected = self._selected_native = False
            reset_item = getattr(owner._kernel, '_reset_item_cache', None)
            if reset_item is None:  # Native/older artifacts keep their root protocol.
                owner._kernel.reset()
            else:
                reset_item()
                self._selected = selected
                self._selected_native = native
