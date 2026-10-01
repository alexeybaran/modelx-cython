"""Lightweight owner used by original generated Space classes.

No Space facade, reference discovery or formula execution lives here. Input
updates retain existing cache semantics; they do not silently invalidate cells.
"""
from importlib import import_module


class SpaceFrameOwner:
    def __init__(self, module, inputs):
        self.module = module
        self.inputs = inputs
        self.kernel = import_module(module).Kernel((inputs,))
        self.kernel.select_scope(0)

    def call(self, cell, args, inputs):
        self.inputs = inputs
        self.kernel.bind_inputs(inputs)
        return self.kernel.call(cell, args)

    def reset(self):
        self.kernel.reset()

    def __reduce__(self):
        return _restore, (self.module, self.inputs, self.kernel.snapshot())


def _restore(module, inputs, entries):
    owner = SpaceFrameOwner(module, inputs)
    owner.kernel.restore(entries)
    return owner
