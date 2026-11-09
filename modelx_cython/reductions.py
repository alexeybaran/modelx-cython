"""Compatibility bridge for the retired AST generator/reduction pass.

Explicit helper calls bind to declared Cython providers. Cython owns ordinary
source generators and callbacks. This bridge never analyzes formula bodies and
never reports a native lowering; old imports can transition independently.
"""
import libcst as cst


class NativeReductions(cst.CSTTransformer):
    def __init__(self, *args, **kwargs):
        self.count = 0
        self.fallbacks = []
        self.exact_receiver_guards = ()
        self.scope_safe = False
