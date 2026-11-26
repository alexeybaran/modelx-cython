"""Lower lexically resolved calls to the ordinary guarded round helper."""
import libcst as cst

from .consts import MX_SYS_MOD
from .scopes import OPAQUE_SCOPES
from .syntax import arguments


ROUND_BINDING = """\
import builtins as _mx_round_builtins
from types import BuiltinFunctionType as _mx_round_builtin_type
_mx_canonical_round = _mx_round_builtins.round
if (type(_mx_canonical_round) is not _mx_round_builtin_type
        or _mx_canonical_round.__self__ is not _mx_round_builtins
        or _mx_canonical_round.__name__ != 'round'):
    _mx_canonical_round = None
"""


class NativeRounding(cst.CSTTransformer):
    def __init__(self):
        self.used = False

    def on_visit(self, node):
        return False if isinstance(node, OPAQUE_SCOPES) else super().on_visit(node)

    def leave_Call(self, original, updated):
        args = arguments(original, 'round')
        if args is None or len(args) not in (1, 2):
            return updated
        self.used = True
        operation = '_mx_round_one' if len(args) == 1 else '_mx_round_digits'
        return updated.with_changes(
            func=cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name(operation)),
            args=(cst.Arg(updated.func), *updated.args, cst.Arg(cst.Name('_mx_canonical_round'))))
