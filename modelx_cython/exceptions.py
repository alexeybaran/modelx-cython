"""Use conditional error checks on generated numeric Cython calls."""
import libcst as cst

from .consts import CY_MOD
from .syntax import dotted_name


_NUMERIC_TYPES = {f'{CY_MOD}.double': 'double', f'{CY_MOD}.longlong': 'long long',
                  f'{CY_MOD}.bint': 'bint'}


def numeric_exception_spec(return_type):
    return ' except? -1' if return_type in _NUMERIC_TYPES.values() else ''


class NumericExceptions(cst.CSTTransformer):
    def leave_FunctionDef(self, before, after):
        decorators = [d.decorator for d in after.decorators]
        if (after.returns and dotted_name(after.returns.annotation) in _NUMERIC_TYPES
                and any(dotted_name(d) in (f'{CY_MOD}.cfunc', f'{CY_MOD}.ccall') for d in decorators)
                and not any(isinstance(d, cst.Call) and dotted_name(d.func) == f'{CY_MOD}.exceptval'
                            for d in decorators)):
            return after.with_changes(decorators=(*after.decorators,
                cst.Decorator(cst.parse_expression(f'{CY_MOD}.exceptval(-1, check=True)'))))
        return after
