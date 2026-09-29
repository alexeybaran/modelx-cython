"""One manifest-driven modelx-shaped facade, independent of any insurance product.

Input replacement commits pandas and packed views together. ItemSpaces own their
compiled caches; the facade never keeps a Python-side Cell-result cache.
"""
from pathlib import Path
import importlib
import inspect
import json
import hashlib
import threading
from collections import OrderedDict
from copy import deepcopy
from .contracts import ContractError
from .data_binding import pack_tables, prepare_inputs
from .runtime import CompiledKernel


class CollectResult:
    """Attribute-style container matching the exported collect result shape."""

    __slots__ = ('_fields',)

    def __init__(self, fields):
        self._fields = fields

    def __getattr__(self, name):
        try:
            return self._fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._fields))

    def __repr__(self):
        return 'CollectResult(%s)' % ', '.join(self._fields)


def _detach_collect_value(value):
    """Copy outputs before a policy's working frame can be selected again."""
    if isinstance(value, (_ItemSpace, _ProjectionSpace)):
        raise TypeError('collect output cannot retain a live Space')
    if type(value) is dict:
        return {key: _detach_collect_value(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_detach_collect_value(item) for item in value)
    return deepcopy(value)


def _collect_shape(value):
    return {name: _collect_shape(item) for name, item in value.items()} if type(value) is dict else None


def _check_collect_shape(expected, value, path):
    if expected is None:
        if type(value) is dict:
            raise ValueError('collect output structure changed at ' + repr(path))
        return
    if type(value) is not dict or tuple(value) != tuple(expected):
        raise ValueError('collect output structure changed at ' + repr(path))
    for name, item in value.items():
        _check_collect_shape(expected[name], item, path + (name,))


def _collect_add(old, new, path):
    if type(new) is dict:
        if old is None:
            old = {name: None for name in new}
        elif type(old) is not dict or tuple(old) != tuple(new):
            raise ValueError('collect output structure changed at ' + repr(path))
        for name, item in new.items():
            old[name] = _collect_add(old[name], item, path + (name,))
        return old
    if type(old) is dict:
        raise ValueError('collect output structure changed at ' + repr(path))
    if old is None:
        return new
    try:
        return old + new
    except Exception as exc:
        raise TypeError('collect sum value cannot be added at ' + repr(path)) from exc


class _CollectAccumulator:
    def __init__(self):
        self.modes = None
        self.individual_shape = None
        self.result = None

    def add(self, coordinate, value):
        if type(value) is not dict or not value or set(value) - {'individual', 'sum'}:
            raise ValueError('collect_outputs must declare individual and/or sum')
        if self.modes is None:
            self.modes = tuple(value)
            self.result = {mode: None for mode in self.modes}
        elif tuple(value) != self.modes:
            raise ValueError('collect_outputs modes changed between policies')
        for mode, items in value.items():
            if type(items) is not dict or not items or any(type(name) is not str for name in items):
                raise TypeError('collect_outputs sections need named values')
            if mode == 'individual':
                shape = {name: _collect_shape(item) for name, item in items.items()}
                if self.individual_shape is None:
                    self.individual_shape = shape
                    self.result[mode] = OrderedDict()
                else:
                    _check_collect_shape(self.individual_shape, items, (mode,))
                self.result[mode][coordinate] = CollectResult(items)
            else:
                self.result[mode] = _collect_add(self.result[mode], items, (mode,))

    def finish(self):
        if self.result is None:
            raise ValueError('collect domain must not be empty')
        return CollectResult({mode: value if mode == 'individual' else CollectResult(value)
                              for mode, value in self.result.items()})


class DataView:
    """Snapshot receiver for ordinary Python formulas using data.<table>()."""
    def __init__(self, tables, reference_names=()):
        object.__setattr__(self, '_tables', tables)
        object.__setattr__(self, '_prepared', {})
        object.__setattr__(self, '_reference_names', frozenset(reference_names))

    def _prepare_lookup(self, name, factory=None):
        from .table_lookup import IndexedTable
        factory = factory or IndexedTable
        key = (name, factory)
        if key not in self._prepared:
            self._prepared[key] = factory(self._tables[name])
        return self._prepared[key]

    def __getattr__(self, name):
        if name in self._tables:
            if name in self._reference_names:
                return self._tables[name]
            def reader():
                return self._tables[name]
            # Bind a reader once, not a new Python closure on every formula call.
            # Read through self so public Data-reference replacement stays live.
            object.__setattr__(self, name, reader)
            return reader
        if name.endswith('_data') and name[:-5] in self._tables: return self._tables[name[:-5]]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError('Formula Data receivers are read-only; replace model Data references')


class _DataSpace(DataView):
    def __init__(self, model):
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_prepared', {})
        object.__setattr__(self, '_reference_names',
                           frozenset(model._plan.get('data_reference_names', ())))

    @property
    def _tables(self):
        self._model._require_open()
        return self._model._tables

    def __setattr__(self, name, value):
        model = self._model
        if name in self._reference_names:
            model._set_table(name, value)
        elif name.endswith('_data') and name[:-5] in model._plan['tables']:
            model._set_table(name[:-5], value)
        else: raise AttributeError('Set tables through their <name>_data references')


class _ItemSpace:
    def __init__(self, space, arguments, dynamic_refs=None):
        self._space, self._arguments = space, arguments
        self._dynamic_refs = dict(dynamic_refs or {})
        self._context = None
        self._entries = {}

    def _ensure_dynamic_refs(self):
        if self._dynamic_refs is None:
            model = self._space._model
            self._dynamic_refs = (model._space_formula_refs(
                bindings=model._space_bindings, **self._arguments)
                if model._space_bindings is not None else {})

    def __getattr__(self, name):
        model = self._space._model
        model._require_open()
        if name in self._arguments: return self._arguments[name]
        self._ensure_dynamic_refs()
        if name in self._dynamic_refs: return self._dynamic_refs[name]
        if name in model._references: return model._references[name]
        if name in model._plan.get('row_references', {}):
            point = self._arguments[model._plan['point_parameter']]
            return model._tables[model._plan['point_table']].at[
                point, model._plan['row_references'][name]]
        if name in model._plan.get('object_references', {}):
            return model._tables[model._plan['object_references'][name]]
        if name in model.manifest['cells'] or name in model.manifest['outputs']:
            if name not in self._entries:
                self._entries[name] = lambda *args, **kwargs: self._call(name, *args, **kwargs)
            return self._entries[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if not name.startswith('_'):
            raise AttributeError('Select parameters with Projection[...] and set references on the Space')
        object.__setattr__(self, name, value)

    def _call(self, name, *args, **kwargs):
        model = self._space._model
        with model._lock:
            model._require_idle()
            point = self._arguments[model._plan['point_parameter']]
            if point not in model._positions: raise KeyError(point)
            if self._context is None:
                self._ensure_dynamic_refs()
                params = {n: v for n, v in self._arguments.items() if n in model.manifest['parameters']}
                row_inputs = {name: value for name, value in self._dynamic_refs.items()
                              if name in model.manifest['inputs']}
                self._context = model._kernel.fork_item(
                    model._positions[point], input_overrides=row_inputs, **params)
            model._active += 1
            try:
                result = self._context.call(name, *args, **kwargs)
            finally:
                model._active -= 1
            if name in model.manifest['outputs']:
                import pandas as pd
                return pd.DataFrame(result, columns=list(model.manifest['outputs'][name]),
                                    index=pd.Index(range(len(result)), name=model._plan['output_index']))
            return result


class _ProjectionSpace:
    def __init__(self, model):
        object.__setattr__(self, '_model', model)
        defaults = model._plan['parameter_defaults']
        required = set(model._plan.get('required_parameters', ()))
        params = [inspect.Parameter(n, getattr(inspect.Parameter, kind),
                                  default=(inspect.Parameter.empty if n in required
                                           else defaults.get(n, inspect.Parameter.empty)))
                  for n, kind in model._plan['space_parameters']]
        object.__setattr__(self, '_signature', inspect.Signature(params))
        point = model._plan['point_parameter']
        object.__setattr__(self, '_collect_signature',
                           inspect.Signature(p for p in params if p.name != point))
        object.__setattr__(self, '_items', {})

    @property
    def parameters(self):
        return () if self._model._plan.get('static_projection') else tuple(self._signature.parameters)

    @property
    def itemspaces(self):
        return {} if self._model._plan.get('static_projection') else dict(self._items)

    def __getitem__(self, key):
        return self(*(key if isinstance(key, tuple) else (key,)))

    def __call__(self, *args, **kwargs):
        model = self._model
        if model._plan.get('static_projection'):
            raise TypeError('This Space is not parameterized')
        with model._lock:
            model._require_open()
            bound = self._signature.bind(*args, **kwargs); bound.apply_defaults()
            point = bound.arguments[model._plan['point_parameter']]
            if point not in model._positions: raise KeyError(point)
            key = tuple(bound.arguments.values())
            if key not in self._items:
                refs = (model._space_formula_refs(bindings=model._space_bindings,
                        **bound.arguments) if model._space_bindings is not None else None)
                self._items[key] = _ItemSpace(self, dict(bound.arguments), refs)
            return self._items[key]

    def collect(self, *args, **kwargs):
        """Traverse declared scalar sums with the reusable compiled context.

        This is the first frame-backend integration shape. Other collect output
        kinds are rejected by the build contract rather than silently using a
        different formula or cache implementation.
        """
        model = self._model
        plan = model._plan.get('collect')
        if plan is None:
            raise TypeError('this Space has no collect declaration')
        bound = self._collect_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        key = tuple(bound.arguments.values())
        with model._lock:
            model._require_idle()
            if key in model._collect_cache:
                return model._collect_cache[key]
            model._prepare_connected_context(bound.arguments, key)
            defaults = model._plan['parameter_defaults']
            changed = {name: value for name, value in bound.arguments.items()
                       if name in model.manifest['parameters'] and value != defaults[name]}
            model._active += 1
            try:
                if changed:
                    model._kernel.set_parameters(**changed)
                try:
                    if 'object_outputs' in plan:
                        coordinates = tuple(model._tables[model._plan['point_table']].index)
                        accumulator = _CollectAccumulator()
                        model._kernel._run_object_outputs(
                            plan['object_outputs'], _detach_collect_value,
                            lambda i, value: accumulator.add(coordinates[i], value))
                        result = accumulator.finish()
                    else:
                        sums = {}
                        for name in plan['sum']:
                            values = model._kernel.run_scalar(name)
                            total = float(values[0])
                            for value in values[1:]:
                                total += float(value)
                            sums[name] = total
                        result = CollectResult({'sum': CollectResult(sums)})
                finally:
                    if changed:
                        model._kernel.set_parameters(
                            **{name: defaults[name] for name in changed})
            finally:
                model._active -= 1
            model._collect_cache[key] = result
            return result

    def __getattr__(self, name):
        self._model._require_open()
        model=self._model
        if name in model._references: return model._references[name]
        if name in model._plan.get('object_references', {}):
            return model._tables[model._plan['object_references'][name]]
        if model._plan.get('static_projection') and (name in model.manifest['cells'] or name in model.manifest['outputs']):
            with model._lock:
                if () not in self._items:
                    self._items[()]=_ItemSpace(self,{model._plan['point_parameter']:0})
                return getattr(self._items[()],name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self._model._references: self._model._set_reference(name, value)
        elif name in self._model._plan.get('object_references', {}):
            self._model._set_table(self._model._plan['object_references'][name],value)
        else: raise AttributeError(name)


class CompiledModel:
    """Load a derived model artifact using only its manifest and pandas inputs.

    Supports the documented Data/reference/independent ItemSpace subset, not full
    modelx metaprogramming or incremental dependency tracing. Untracked in-place
    DataFrame mutation is unsupported: assign a replacement Data reference.
    """
    def __init__(self, artifact, tables, *, overrides=None, space_bindings=None):
        self.artifact = Path(artifact).resolve()
        self.manifest = json.loads((self.artifact / 'manifest.json').read_text())
        self._plan = self.manifest.get('model', {})
        if self._plan.get('schema') != 'mxcf.model.v1': raise ContractError('A derived model manifest is required')
        digest = hashlib.sha256(json.dumps(self._plan, sort_keys=True).encode()).hexdigest()
        if digest != self.manifest.get('model_sha256'):
            raise ContractError('Model binding contract hash mismatch')
        collect = self._plan.get('collect')
        if collect is not None:
            if type(collect) is not dict:
                raise ContractError('Unsupported compiled collect declaration')
            numeric = ('sum' in collect and type(collect.get('sum')) is list
                       and bool(collect['sum'])
                       and all(type(name) is str and name in self.manifest['scalar_outputs']
                               and self.manifest['cells'][name]['result']['kind'] == 'float'
                               for name in collect['sum'])
                       and len(set(collect['sum'])) == len(collect['sum'])
                       and set(collect) == {'over', 'sum'})
            object_outputs = (collect.get('object_outputs') == 'collect_outputs'
                              and set(collect) == {'over', 'object_outputs'}
                              and collect['object_outputs'] in self.manifest.get('object_cells', ()))
            if (collect.get('over') != self._plan['point_parameter']
                    or not (numeric or object_outputs)):
                raise ContractError('Unsupported compiled collect declaration')
        self._lock, self._closed, self._active = threading.RLock(), False, 0
        self._collect_cache = {}
        self._references = {**self._plan['reference_defaults'], **dict(overrides or {})}
        if set(self._references) != set(self._plan['reference_defaults']): raise ContractError('Unknown reference')
        missing = [name for name in self._plan.get('required_object_bindings', ())
                   if self._references[name] is None]
        if missing:
            raise ContractError('Missing connected object references: ' + repr(missing))
        self._tables = self._derive_tables(self._snapshot(tables))
        self._packed = pack_tables(self._plan, self._tables)
        records = prepare_inputs(self.manifest, self._tables, references=self._references,
                                 packed=self._packed, prepared=True)
        self._kernel = CompiledKernel(self.artifact, records, python_bindings=self._bindings(self._tables, self._references))
        space_formula = self._plan.get('space_formula')
        self._space_preparer = (importlib.import_module(
            self.manifest['package'] + '._space_formula').prepare
            if space_formula is not None else None)
        if space_bindings is not None:
            if self._space_preparer is None:
                raise ContractError('Connected bindings require a compiled Space formula')
            expected = set(space_formula['references'])
            if type(space_bindings) is not dict or set(space_bindings) != expected:
                raise ContractError('Space formula bindings differ from declared references')
        self._space_bindings = (dict(space_bindings) if space_bindings is not None
                                else None)
        self._connected_context_key = None
        self._positions = {key: i for i, key in enumerate(self._tables[self._plan['point_table']].index)}
        self._data, self._projection = _DataSpace(self), _ProjectionSpace(self)

    @staticmethod
    def _snapshot(tables):
        import pandas as pd
        import numpy as np
        if any(not isinstance(t, (pd.DataFrame,pd.Series,np.ndarray)) for t in tables.values()):
            raise ContractError('DataFrames, Series or ndarrays required')
        return {name: table.copy() if isinstance(table,np.ndarray) else table.copy(deep=True)
                for name, table in tables.items()}

    def _derive_tables(self, tables):
        program = self._plan.get('data_program')
        if program is None:
            return tables
        import pandas as pd
        from .data_program import prepare_data
        references = {name: tables[name] for name in program['references']}
        derived, domain = prepare_data(program, references)
        updated = {**tables, **self._snapshot(derived)}
        updated[program['policy_table']] = pd.DataFrame(
            index=pd.Index(domain, name=self._plan['point_parameter']))
        return updated

    def _require_open(self):
        if self._closed: raise RuntimeError('Compiled model is closed')

    def _require_idle(self):
        self._require_open()
        if self._active:
            raise RuntimeError('Cannot change model inputs or reset during an active Cell evaluation')

    def _space_formula_refs(self, *args, bindings, **kwargs):
        """Evaluate the source Space formula with explicit connected references.

        This is an integration hook for connected model preparation, not a
        replacement for ordinary ItemSpace construction or Cell execution.
        """
        if self._space_preparer is None:
            raise ContractError('No compiled Space formula in this artifact')
        expected = set(self._plan['space_formula']['references'])
        if type(bindings) is not dict or set(bindings) != expected:
            raise ContractError('Space formula bindings differ from declared references')
        bound = self._projection._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        with self._lock:
            self._require_idle()
            self._active += 1
            try:
                result = self._space_preparer(bindings, *bound.args)
            finally:
                self._active -= 1
        if (type(result) is not dict or set(result) != {'refs'}
                or type(result['refs']) is not dict
                or any(type(name) is not str or name.startswith('_')
                       for name in result['refs'])):
            raise ValueError('Compiled Space formula must return a refs dictionary')
        return result['refs']

    def _prepare_connected_context(self, arguments, key):
        """Prepare one complete policy domain before its native traversal."""
        if self._space_bindings is None or self._connected_context_key == key:
            return
        point_name = self._plan['point_parameter']
        point_table = self._plan['point_table']
        columns = self._plan.get('row_references', {})
        frame = self._tables[point_table].copy(deep=True)
        expected = None
        for point in frame.index:
            args = {**arguments, point_name: point}
            refs = self._space_formula_refs(bindings=self._space_bindings, **args)
            if expected is None:
                expected = frozenset(refs)
                if not expected <= columns.keys():
                    raise ContractError('Space formula returned references absent from prepared rows')
            elif frozenset(refs) != expected:
                raise ContractError('Space formula reference names vary by policy')
            for name, value in refs.items():
                frame.at[point, columns[name]] = value
        retained = dict(self._collect_cache)
        tables = {**self._tables, point_table: frame}
        self._commit(tables, self._references,
                     pack_tables(self._plan, tables), preserve_item_contexts=True)
        self._collect_cache.update(retained)
        self._connected_context_key = key

    def __getattr__(self, name):
        if name == self._plan['data_space']: return self._data
        if name == self._plan['projection']: return self._projection
        raise AttributeError(name)

    def _bindings(self, tables, refs):
        required = set(self.manifest.get('python_references', ()))
        if not required: return None
        import importlib
        data = DataView(tables, self._plan.get('data_reference_names', ()))
        # Validate/preflight every prepared table before replacing live bindings.
        for name in self.manifest.get('prepared_lookup_tables',()):data._prepare_lookup(name)
        shared = {'data': data, **refs, **self._plan['parameter_defaults'],
                  **{name: tables[table] for name,table in self._plan.get('object_references',{}).items()},
                  **{name: importlib.import_module(module) for name, module in
                     self._plan.get('module_references', {}).items()}}
        rows = []
        for key in tables[self._plan['point_table']].index:
            row = {**shared, self._plan['point_parameter']: key}
            if required - row.keys(): raise ContractError('Unbound formula references: ' + repr(sorted(required - row.keys())))
            rows.append({n: row[n] for n in required})
        return rows

    def _commit(self, tables, refs, packed, *, preserve_item_contexts=False):
        records = prepare_inputs(self.manifest, tables, references=refs,
                                 packed=packed, prepared=True)
        self._kernel.bind(records, python_bindings=self._bindings(tables, refs))
        self._tables, self._references, self._packed = tables, refs, packed
        self._collect_cache.clear()
        self._connected_context_key = None
        self._positions = {key: i for i, key in enumerate(tables[self._plan['point_table']].index)}
        if not preserve_item_contexts:
            for item in self._projection._items.values():
                item._context = None
                item._dynamic_refs = None

    def _set_table(self, name, value):
        with self._lock:
            self._require_idle()
            program = self._plan.get('data_program')
            if program is not None and name not in program['references']:
                raise ContractError('Replace a source Data reference, not a derived Data Cell')
            tables = self._derive_tables({**self._tables, **self._snapshot({name: value})})
            self._commit(tables, self._references, pack_tables(self._plan, tables))

    def _set_reference(self, name, value):
        with self._lock:
            self._require_idle()
            if name in self.manifest['inputs']:
                self._kernel.update_inputs(**{name: value})
            self._references = {**self._references, name: value}
            self._collect_cache.clear()
            for item in self._projection._items.values():
                item._context = None
                item._dynamic_refs = None

    def reset(self):
        """Clear compiled caches while retaining input packing and ItemSpace storage."""
        with self._lock:
            self._require_idle()
            self._kernel.reset()
            self._collect_cache.clear()
            self._connected_context_key = None
            for item in self._projection._items.values():
                if self._space_bindings is not None:
                    item._context = None
                    item._dynamic_refs = None
                elif item._context is not None:
                    item._context.reset()

    def close(self):
        with self._lock:
            if self._closed: return
            self.reset()
            for item in self._projection._items.values(): item._context = None
            self._closed = True
            self._kernel = None
            self._tables, self._packed = {}, {}
            self._projection._items.clear()


def _load_embedded_model(package_file):
    """Instantiate the model exported by a generated compiled package.

    This is intentionally private. Generated ``Model_nomx_cy.__init__`` modules
    call it so normal users import a ready model object instead of constructing
    ``CompiledModel(artifact, tables)`` themselves.
    """
    import pickle

    artifact = Path(package_file).resolve().parent
    manifest = json.loads((artifact / 'manifest.json').read_text())
    public = manifest.get('public_api', {})
    if public.get('schema') != 'mxcf.package.v1':
        raise ContractError('Compiled artifact has no supported public model package')
    path = artifact / public['tables_file']
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != public.get('tables_sha256'):
        raise ContractError('Embedded model Data snapshot hash mismatch')
    tables = pickle.loads(payload)
    return CompiledModel(artifact, tables)
