"""Test-only historical facade assembly around the current private emitter."""
from modelx_cython._frame_runtime.compiler import *
from modelx_cython._frame_runtime.compiler import build as _build
from pathlib import Path
import json

def build(spec, destination, **kwargs):
    result = _build(spec, destination, **kwargs)
    if spec.model is not None and spec.public_emission_plan is None and not kwargs.get('package_build'):
        from .package_codegen import add_public_package
        p = Path(destination) / 'manifest.json'
        manifest = json.loads(p.read_text())
        add_public_package(destination, manifest, spec.initial_tables, spec.model_name)
        p.write_text(json.dumps(manifest, indent=2))
    return result
