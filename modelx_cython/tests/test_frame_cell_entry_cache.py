"""Compiled entry code can be shared without sharing Cell receiver state."""

from modelx_cython._frame_runtime.python_calculator import (
    CellCodec, _compiled_cell_entry,
)


class _Kernel:
    def __init__(self, base):
        self.base = base
        self.calls = []

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        return self.base + arguments[0]


def test_reused_entry_code_keeps_separate_receiver_and_defaults():
    manifest = {
        'inputs': {},
        'cells': {'value': {
            'params': [{'name': 'offset', 'default': '3',
                        'type': {'kind': 'int'}}],
            'result': {'kind': 'int'},
        }},
    }
    _compiled_cell_entry.cache_clear()
    first_kernel, second_kernel = _Kernel(10), _Kernel(20)
    first = CellCodec(manifest, first_kernel, ())
    second = CellCodec(manifest, second_kernel, ())

    assert first.entry('value')() == 13
    assert second.entry('value')(offset=4) == 24
    assert first.entry('value')(5) == 15
    assert first_kernel.calls == [('value', (3,)), ('value', (5,))]
    assert second_kernel.calls == [('value', (4,))]
    info = _compiled_cell_entry.cache_info()
    assert (info.misses, info.hits) == (1, 1)
