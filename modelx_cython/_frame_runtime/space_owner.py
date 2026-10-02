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
        self._extension, self._manifest = _metadata(module)
        self.kernel = self._extension.Kernel((inputs,))
        self._scope = None
        if self._manifest and self._manifest.get('python_cells'):
            self.kernel.select_native_scope(0)
        else:
            self.kernel.select_scope(0)

    def prepare_checked(self):
        if self._scope is not None or not self._manifest.get('python_cells'):
            return
        from ._mx_frame_support.python_calculator import PythonScope
        scope = PythonScope(None, self._manifest, self.kernel,
                            program=_binding_program(self.module))
        self.kernel.set_python_scope(scope)
        self.kernel.attach_python_scope_current()
        self._scope = scope

    def entry_capsule(self, cell):
        """Bind the private typed ABI after the ordinary lazy kernel load."""
        return self._extension.__pyx_capi__['_mx_shell_' + cell]

    def virtual_capsule(self, cell):
        """Checked cache entry; a miss invokes the original virtual formula."""
        return self._extension.__pyx_capi__['_mx_virtual_' + cell]

    def call(self, cell, args, inputs):
        self.prepare_checked()
        self.kernel.bind_inputs(inputs)
        return self.kernel.call(cell, args)

    @property
    def inputs(self):
        return self.kernel.bound_inputs()

    def reset(self):
        self.kernel._reset_item_cache()

    def __reduce__(self):
        return _restore, (self.module, self.inputs, self.kernel.snapshot())


def _restore(module, inputs, entries):
    owner = SpaceFrameOwner(module, inputs)
    if owner._manifest.get('python_cells'):
        owner.prepare_checked()
    owner.kernel.restore(entries)
    return owner
