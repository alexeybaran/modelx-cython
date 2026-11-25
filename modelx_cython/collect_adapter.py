"""Recognize the exporter-owned collection evaluator without changing it."""

import libcst as cst

from .syntax import dotted_name, literal_value, statements


_VERSION = '_mx_collect_adapter_version'
PROTOCOL = 2
_HOOKS = {
    'collect_from_space': {'evaluate_serial', 'reuse_in_parallel', 'private_ordered'},
    '_mx_collect_parallel': {'private', 'private_ordered', 'workers'},
    '_mx_collect_chunk': {'private'},
    '_mx_collect_native_coords': set(),
}


def _current(tree):
    declarations = []
    for node in statements(tree):
        if isinstance(node, cst.Assign) and any(
                dotted_name(target.target) == _VERSION for target in node.targets):
            declarations.append(node.value)
        elif isinstance(node, (cst.AnnAssign, cst.AugAssign)) and dotted_name(node.target) == _VERSION:
            return False
    if len(declarations) != 1:
        return False
    try:
        version = literal_value(declarations[0])
    except (ValueError, TypeError):
        return False
    if type(version) is not int or version != PROTOCOL:
        return False
    for name, arguments in _HOOKS.items():
        functions = [node for node in tree.body
                     if isinstance(node, cst.FunctionDef) and node.name.value == name]
        if (len(functions) != 1 or functions[0].asynchronous is not None
                or not arguments <= {param.name.value for param in functions[0].params.params}):
            return False
    return True


def _flag(tree, name):
    for node in statements(tree):
        if isinstance(node, cst.Assign) and any(dotted_name(target.target) == name for target in node.targets):
            try:
                return literal_value(node.value) is True
            except (ValueError, TypeError):
                return False
    return False


class CollectRuntime:
    def __init__(self, source):
        self.syntax = cst.parse_module(source)
        self.hook = _current(self.syntax)
        # protocol 2 extension: the evaluator may take the collect's other Space parameters
        self.evaluator_args = self.hook and _flag(self.syntax, '_mx_collect_evaluator_args')
        self.ordered_rows = self.hook and _flag(self.syntax, '_mx_collect_ordered_rows_v1')
        self.packed_sums = self.hook and _flag(self.syntax, '_mx_collect_packed_sums_v1')

    def require_current(self, modules=(), workers=1):
        """The check of the modelx that wrote the export: a build that uses collect (a Space with a
        ``collect`` method or collect cache, a collect option in the spec) or parallel workers needs
        export protocol 2 with its collect entries in ``_mx_sys.py``. A build that uses neither
        compiles an export of any modelx, as upstream does."""
        if not self.hook and (workers > 1 or any(_uses_collect(module) for module in modules)):
            raise ValueError(
                f'This export was written by a modelx without export protocol {PROTOCOL} '
                f'({_VERSION} = {PROTOCOL} in _mx_sys.py, with its collect entries), which '
                f'modelx-cython requires for collect and parallel workers. Re-export the model with a '
                f'modelx that writes protocol {PROTOCOL}.')


_COLLECT_OPTIONS = ('serial_collect', 'parallel_collect_reuse', 'parallel_collect_order',
                    'factory_collect_reuse', 'param_bound_checks')


def _uses_collect(module):
    """A module whose Spaces define ``collect`` or a collect cache, or whose spec sets a collect option."""
    if getattr(getattr(module, 'visitor', None), 'collect_cache_classes', None):
        return True
    for cls in getattr(module, 'classes', {}).values():
        if any(option in module.spec.get_spec(cls.fqname) for option in _COLLECT_OPTIONS):
            return True
    from .syntax import walk
    return any(isinstance(node, cst.FunctionDef) and node.name.value == 'collect'
               for node in walk(module.visitor.wrapper.module))
