"""Capture builtin extrema inputs inside Cython, without Python boxing.

Cython's own optimizer chooses the eligible calls and performs comparison/type
selection. Its first input may otherwise be read after later arguments, which
can change the result when those arguments mutate a native field. Materialize
the inputs with Cython's existing temporary nodes before that transformation.
This affects build time only; generated models need no additional runtime.
"""
from contextlib import contextmanager


@contextmanager
def capture_extrema_inputs():
    from Cython.Compiler.Optimize import EarlyReplaceBuiltinCalls
    from Cython.Compiler import UtilNodes

    original = EarlyReplaceBuiltinCalls._optimise_min_max

    def captured(self, node, args, operator):
        if len(args) == 1 and args[0].is_sequence_constructor:
            args = args[0].args
        if len(args) <= 1:
            return original(self, node, args, operator)
        # Cython already captures args[1:] in order. Capture only the missing
        # first input before delegating; do not duplicate its remaining temps.
        first = UtilNodes.ResultRefNode(args[0])
        result = original(self, node, [first, *args[1:]], operator)
        return UtilNodes.EvalWithTempExprNode(first, result)

    EarlyReplaceBuiltinCalls._optimise_min_max = captured
    try:
        yield
    finally:
        EarlyReplaceBuiltinCalls._optimise_min_max = original
