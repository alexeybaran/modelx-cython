"""Lower calls to recognized scalar builtins with the original callable fallback."""

import libcst as cst

from .consts import MX_SYS_MOD
from .powers import _OPAQUE_SCOPES


CANONICAL_BINDINGS = """\
import builtins as _mx_native_builtins
import math as _mx_native_math
from types import BuiltinFunctionType as _mx_native_builtin_type

def _mx_native_canonical(module, name):
    fn = getattr(module, name, None)
    if (type(fn) is _mx_native_builtin_type and fn.__self__ is module
            and fn.__module__ == module.__name__ and fn.__name__ == name):
        return fn
    return None

_mx_canonical_round = _mx_native_canonical(_mx_native_builtins, 'round')
_mx_canonical_floor = _mx_native_canonical(_mx_native_math, 'floor')
_mx_canonical_ceil = _mx_native_canonical(_mx_native_math, 'ceil')
"""


class NativeMath(cst.CSTTransformer):
    def __init__(self):
        self.used = False

    def on_visit(self, node):
        if isinstance(node, _OPAQUE_SCOPES):
            return False
        return super().on_visit(node)

    def leave_Call(self, original, updated):
        if any(arg.keyword or arg.star for arg in original.args):
            return updated
        fn = original.func
        extra = []
        if isinstance(fn, cst.Name) and fn.value == 'round' and len(original.args) in (1, 2):
            name = 'round'
            operation = '_mx_round_one' if len(original.args) == 1 else '_mx_round_digits'
        elif (isinstance(fn, cst.Attribute) and fn.attr.value in ('floor', 'ceil')
              and len(original.args) == 1 and self._math_reference(fn.value)):
            name = fn.attr.value
            operation = '_mx_integral_math_call'
            extra.append(cst.Arg(cst.Name('True' if name == 'ceil' else 'False')))
        else:
            return updated
        self.used = True
        canonical = cst.Arg(cst.Name('_mx_canonical_' + name))
        return updated.with_changes(
            func=cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name(operation)),
            args=(cst.Arg(updated.func), *updated.args, canonical, *extra),
        )

    @staticmethod
    def _math_reference(node):
        return (isinstance(node, cst.Name) and node.value == 'math'
                or isinstance(node, cst.Attribute) and node.attr.value == 'math'
                and isinstance(node.value, cst.Name) and node.value.value == 'self')
