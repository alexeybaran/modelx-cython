"""Lightweight owner used by original generated Space classes.

No Space facade, reference discovery or formula execution lives here. Input
updates retain existing cache semantics; they do not silently invalidate cells.
"""
from importlib import import_module
import hashlib
import json
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=None)
def _metadata(module):
    extension = import_module(module)
    path = Path(extension.__file__).parent.parent / 'manifest.json'
    return extension, json.loads(path.read_text(encoding='utf-8'))


@lru_cache(maxsize=None)
def _binding_program(module):
    extension, manifest = _metadata(module)
    path = Path(extension.__file__).parent / '_python.py'
    source = path.read_text(encoding='utf-8')
    if hashlib.sha256(source.encode()).hexdigest() != manifest['python_source_sha256']:
        raise ValueError('Generated binding source hash mismatch')
    return compile(source, str(path), 'exec')


class SpaceFrameOwner:
    def __init__(self, module, inputs):
        self.module = module
        self._prepared_record = inputs
        self._extension, self._manifest = _metadata(module)
        self.kernel = self._extension.Kernel((inputs,))
        self._scope = None
        if self._manifest and self._manifest.get('python_cells'):
            self.kernel.select_native_scope(0)
        else:
            self.kernel.select_scope(0)

    def prepare_checked(self, bindings=None):
        if self._scope is not None or not self._manifest.get('python_cells'):
            return
        if __package__.endswith('._frame_runtime'):
            from .python_calculator import PythonScope
        else:
            from ._mx_frame_support.python_calculator import PythonScope
        scope = PythonScope(None, self._manifest, self.kernel, bindings,
                            program=_binding_program(self.module))
        self.kernel.set_python_scope(scope)
        if self._manifest.get('public_entry_abi'):
            # First attachment, before any formula is executed. A prepared
            # record needs physical decoding, not bind_original_inputs.
            self.kernel.select_scope(0)
        else:
            self.kernel.attach_python_scope_current()
        self._scope = scope

    def entry_capsule(self, cell):
        """Bind the private typed ABI after the ordinary lazy kernel load."""
        prefix = '_mx_public_' if self._manifest.get('public_entry_abi') else '_mx_shell_'
        return self._extension.__pyx_capi__[prefix + cell]

    def virtual_capsule(self, cell):
        """Checked cache entry; a miss invokes the original virtual formula."""
        return self._extension.__pyx_capi__['_mx_virtual_' + cell]

    def call(self, cell, args, inputs):
        self.prepare_checked()
        self.kernel.bind_inputs(inputs)
        return self.kernel.call(cell, args)

    @property
    def inputs(self):
        if self._manifest.get('public_entry_abi'):
            return self._prepared_record
        return self.kernel.bound_inputs()

    def reset(self):
        self.kernel._reset_item_cache()

    def __reduce__(self):
        if self._manifest.get('public_entry_abi'):
            bindings = self._scope.bindings if self._scope is not None else None
            return _restore_prepared, (self.module, self.inputs, bindings, self.kernel.snapshot())
        return _restore, (self.module, self.inputs, self.kernel.snapshot())


def prepared_owner(receiver, module, resources, existing=None):
    """Prepare resolved resources behind original classes, never a Space facade.

    This is the initial-owner boundary. Replacement/synchronization is a
    separate responsibility; this helper does not invalidate completed Cells.
    The generated public-binding descriptor resolves aliases at compile time.
    """
    extension, manifest = _metadata(module)
    binding = manifest['public_space_binding']
    if module not in resources:
        if __package__.endswith('._frame_runtime'):
            from .prepared_space_inputs import PreparedSpaceInputs
            from .data_binding import DataView
        else:
            from ._mx_frame_support.prepared_space_inputs import PreparedSpaceInputs
            from ._mx_frame_support.data_binding import DataView
        plan = manifest['model']
        data = getattr(receiver, binding['data_reference'])
        reference_names = frozenset(plan.get('data_reference_names', ()))
        tables = {name: (getattr(data, name) if name in reference_names
                         else getattr(data, name)()) for name in plan['tables']
                  if name != binding.get('synthetic_policy_table')}
        if 'synthetic_policy_table' in binding:
            # This index is preparation metadata, not an extra economic Data
            # Cell. Its existing formula executes on the original compiled
            # receiver; never reconstruct or interpret the author formula.
            import pandas as pd
            domain = tuple(getattr(receiver, binding['domain_entry'])())
            schema = plan['tables'][binding['synthetic_policy_table']]
            tables[binding['synthetic_policy_table']] = pd.DataFrame(
                index=pd.Index(domain, name=schema['index_names'][0]))
        # Original public Data Cells may expose Cython memoryviews where the
        # physical preparation schema describes their underlying ndarray.
        # np.asarray borrows that view here; established indexed-input packing
        # subsequently applies its declared snapshot contract. Live mutable
        # array synchronization is a separate qualification gate.
        import numpy as np
        for name, schema in plan['tables'].items():
            if schema.get('object_kind') == 'ndarray' and not isinstance(tables[name], np.ndarray):
                tables[name] = np.asarray(tables[name])
        references = {name: getattr(receiver, name) for name in plan['reference_defaults']}
        lookups = import_module(module.rsplit('.', 1)[0] + '._lookups')
        prepared = PreparedSpaceInputs(manifest, lookups, tables, references=references)
        point = tables[plan['point_table']] if plan['point_table'] is not None else None
        positions = {label: i for i, label in enumerate(point.index)} if point is not None else {0: 0}
        resources[module] = prepared, positions, DataView(tables, reference_names)
    prepared, positions, data_view = resources[module]
    plan = manifest['model']
    label = getattr(receiver, plan['point_parameter']) if plan['point_parameter'] else 0
    parameters = {name: getattr(receiver, name) for name in plan['parameter_defaults']}
    record = prepared.record(positions[label], parameters=parameters)
    bindings = {name: (data_view if name == 'data' else getattr(receiver, name))
                for name in manifest.get('python_references', ())}
    if existing is None:
        owner = SpaceFrameOwner(module, record)
        owner.prepare_checked((bindings,))
    else:
        if existing.module != module:
            raise ValueError('Collection owner has a different generated layout')
        owner = existing
        if owner._scope is not None:
            owner._scope.bindings = (bindings,)
        owner.kernel.replace_records((record,))
        owner.kernel.select_scope(0)
        owner._prepared_record = record
    return owner


def _restore(module, inputs, entries):
    owner = SpaceFrameOwner(module, inputs)
    if owner._manifest.get('python_cells'):
        owner.prepare_checked()
    owner.kernel.restore(entries)
    return owner


def _restore_prepared(module, inputs, bindings, entries):
    owner = SpaceFrameOwner(module, inputs)
    owner.prepare_checked(bindings)
    owner.kernel.restore(entries)
    return owner
