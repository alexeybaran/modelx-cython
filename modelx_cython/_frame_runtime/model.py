"""One manifest-driven modelx-shaped facade, independent of any insurance product.

Input replacement commits pandas and packed views together. ItemSpaces own their
compiled caches; the facade never keeps a Python-side Cell-result cache.
"""
from pathlib import Path
import inspect
import json
import hashlib
import threading
from .contracts import ContractError
from .data_binding import pack_tables, prepare_inputs
from .runtime import CompiledKernel


class DataView:
    """Snapshot receiver for ordinary Python formulas using data.<table>()."""
    def __init__(self, tables):
        object.__setattr__(self, '_tables', tables)
        object.__setattr__(self, '_prepared', {})

    def _prepare_lookup(self, name, factory=None):
        from .table_lookup import IndexedTable
        factory = factory or IndexedTable
        key = (name, factory)
        if key not in self._prepared:
            self._prepared[key] = factory(self._tables[name])
        return self._prepared[key]

    def __getattr__(self, name):
        if name in self._tables:
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

    @property
    def _tables(self):
        self._model._require_open()
        return self._model._tables

    def __setattr__(self, name, value):
        model = self._model
        if name.endswith('_data') and name[:-5] in model._plan['tables']:
            model._set_table(name[:-5], value)
        else: raise AttributeError('Set tables through their <name>_data references')


class _ItemSpace:
    def __init__(self, space, arguments):
        self._space, self._arguments = space, arguments
        self._context = None
        self._entries = {}

    def __getattr__(self, name):
        model = self._space._model
        model._require_open()
        if name in self._arguments: return self._arguments[name]
        if name in model._references: return model._references[name]
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
                params = {n: v for n, v in self._arguments.items() if n in model.manifest['parameters']}
                self._context = model._kernel.fork_item(model._positions[point], **params)
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
        params = [inspect.Parameter(n, getattr(inspect.Parameter, kind),
                                  default=defaults.get(n, inspect.Parameter.empty))
                  for n, kind in model._plan['space_parameters']]
        object.__setattr__(self, '_signature', inspect.Signature(params))
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
            if key not in self._items: self._items[key] = _ItemSpace(self, dict(bound.arguments))
            return self._items[key]

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
    def __init__(self, artifact, tables, *, overrides=None):
        self.artifact = Path(artifact).resolve()
        self.manifest = json.loads((self.artifact / 'manifest.json').read_text())
        self._plan = self.manifest.get('model', {})
        if self._plan.get('schema') != 'mxcf.model.v1': raise ContractError('A derived model manifest is required')
        digest = hashlib.sha256(json.dumps(self._plan, sort_keys=True).encode()).hexdigest()
        if digest != self.manifest.get('model_sha256'):
            raise ContractError('Model binding contract hash mismatch')
        self._lock, self._closed, self._active = threading.RLock(), False, 0
        self._references = {**self._plan['reference_defaults'], **dict(overrides or {})}
        if set(self._references) != set(self._plan['reference_defaults']): raise ContractError('Unknown reference')
        self._tables = self._snapshot(tables)
        self._packed = pack_tables(self._plan, self._tables)
        records = prepare_inputs(self.manifest, self._tables, references=self._references, packed=self._packed)
        self._kernel = CompiledKernel(self.artifact, records, python_bindings=self._bindings(self._tables, self._references))
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

    def _require_open(self):
        if self._closed: raise RuntimeError('Compiled model is closed')

    def _require_idle(self):
        self._require_open()
        if self._active:
            raise RuntimeError('Cannot change model inputs or reset during an active Cell evaluation')

    def __getattr__(self, name):
        if name == self._plan['data_space']: return self._data
        if name == self._plan['projection']: return self._projection
        raise AttributeError(name)

    def _bindings(self, tables, refs):
        required = set(self.manifest.get('python_references', ()))
        if not required: return None
        data = DataView(tables)
        # Validate/preflight every prepared table before replacing live bindings.
        for name in self.manifest.get('prepared_lookup_tables',()):data._prepare_lookup(name)
        shared = {'data': data, **refs, **self._plan['parameter_defaults'],
                  **{name: tables[table] for name,table in self._plan.get('object_references',{}).items()}}
        rows = []
        for key in tables[self._plan['point_table']].index:
            row = {**shared, self._plan['point_parameter']: key}
            if required - row.keys(): raise ContractError('Unbound formula references: ' + repr(sorted(required - row.keys())))
            rows.append({n: row[n] for n in required})
        return rows

    def _commit(self, tables, refs, packed):
        records = prepare_inputs(self.manifest, tables, references=refs, packed=packed)
        self._kernel.bind(records, python_bindings=self._bindings(tables, refs))
        self._tables, self._references, self._packed = tables, refs, packed
        self._positions = {key: i for i, key in enumerate(tables[self._plan['point_table']].index)}
        for item in self._projection._items.values(): item._context = None

    def _set_table(self, name, value):
        with self._lock:
            self._require_idle()
            tables = {**self._tables, **self._snapshot({name: value})}
            self._commit(tables, self._references, pack_tables(self._plan, tables))

    def _set_reference(self, name, value):
        with self._lock:
            self._require_idle()
            if name in self.manifest['inputs']:
                self._kernel.update_inputs(**{name: value})
            self._references = {**self._references, name: value}
            for item in self._projection._items.values(): item._context = None

    def reset(self):
        """Clear compiled caches while retaining input packing and ItemSpace storage."""
        with self._lock:
            self._require_idle()
            self._kernel.reset()
            for item in self._projection._items.values():
                if item._context is not None: item._context.reset()

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
