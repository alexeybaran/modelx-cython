"""The structured build emitter preserves the generated frame program."""
import hashlib
from types import ModuleType

import pytest

from modelx_cython._frame_runtime.compiler import emit
from modelx_cython._frame_runtime.contracts import KernelSpec, cell
from modelx_cython._frame_runtime.frame_codegen import to_c_frame


@cell
def value(t: int) -> float:
    return base + 0.5 * t


@cell
def total() -> float:
    return value(1) + value(2)


@cell
def report() -> object:
    return {'total': total()}


@pytest.mark.parametrize('mixed, expected', [
    (False, '9d940b287c2057d29532f9e9ec2602aaa1fcd75f87995d3027dc94516db5c6af'),
    (True, '54d8bcaefe6d59ae9d1389e265bf911fad8dd5217fc8c888d9042a5b11c6a2d9'),
])
def test_structured_emission_preserves_generated_source(mixed, expected):
    module = ModuleType('frame_emission_fixture')
    module.value = value
    module.total = total
    if mixed:
        module.report = report
    spec = KernelSpec(module=module, inputs={'base': float}, outputs={},
                      steps=4, length='4',
                      cell_outputs=('report',) if mixed else ('total',),
                      scalar_outputs=('total',), cython_objects=mixed)
    source, old_manifest = emit(spec)
    parts, manifest = emit(spec, frame_parts=True)
    assert source.startswith('# cython:')
    assert old_manifest == manifest
    direct = to_c_frame(parts, manifest)
    assert hashlib.sha256(direct.encode()).hexdigest() == expected
