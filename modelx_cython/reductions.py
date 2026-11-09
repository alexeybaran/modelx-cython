"""Compatibility bridge for the retired AST generator/reduction pass.

Explicit helper calls bind to declared Cython providers. Cython owns ordinary
source generators and callbacks. This bridge never analyzes formula bodies and
never reports a native lowering; old imports can transition independently.
"""
import libcst as cst
from .syntax import arguments, literal_value
from .typedefs import primitive_kind


def safe_range_increment(iterator):
    """Prove the final C loop increment, including an overshoot of stop."""
    args = arguments(iterator)
    if args is None:
        return False
    if len(args) < 3:
        return True
    try:
        step = literal_value(args[2])
        if type(step) not in (int, bool):
            return False
        if step in (-1, 1):
            return True
        bounds = [literal_value(arg) for arg in args]
        if not all(primitive_kind(value) in ('longlong', 'bint') for value in bounds):
            return False
        values = range(*bounds)
        return not values or primitive_kind(values[-1] + step) == 'longlong'
    except (ValueError, TypeError):
        return False



class NativeReductions(cst.CSTTransformer):
    def __init__(self, *args, **kwargs):
        self.count = 0
        self.fallbacks = []
        self.exact_receiver_guards = ()
        self.scope_safe = False
