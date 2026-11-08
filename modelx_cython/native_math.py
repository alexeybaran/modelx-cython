"""Lower calls to recognized scalar builtins with the original callable fallback."""

import builtins
import math

import libcst as cst

from .consts import MX_SYS_MOD
from .scopes import OPAQUE_SCOPES


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


def callable_operation(value):
    """Trace a callable's capability; runtime identity is still checked."""
    for name, original in (('round', builtins.round), ('floor', math.floor), ('ceil', math.ceil)):
        if value is original:
            return name
    return None


class NativeMath(cst.CSTTransformer):
    def __init__(self, refs=None):
        self.used = False
        self.refs = refs or {}

    def on_visit(self, node):
        if isinstance(node, OPAQUE_SCOPES):
            return False
        return super().on_visit(node)

    def leave_Call(self, original, updated):
        if any(arg.keyword or arg.star for arg in original.args):
            return updated
        fn = original.func
        extra = []
        name = None
        if isinstance(fn, cst.Name) and fn.value == 'round':
            name = 'round'
        elif isinstance(fn, cst.Attribute):
            if isinstance(fn.value, cst.Name) and fn.value.value == 'self':
                name = getattr(self.refs.get(fn.attr.value), 'math_operation', None)
            if name is None and fn.attr.value in ('floor', 'ceil'):
                # Any module alias or object method is safe to offer: the
                # runtime checks the actual callable and preserves fallback.
                name = fn.attr.value
        if name == 'round' and len(original.args) in (1, 2):
            operation = '_mx_round_one' if len(original.args) == 1 else '_mx_round_digits'
        elif name in ('floor', 'ceil') and len(original.args) == 1:
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
