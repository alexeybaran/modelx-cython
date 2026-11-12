"""Recognize the exporter-owned collection evaluator without changing it."""

import libcst as cst

from .syntax import dotted_name, literal_value, statements, walk


_VERSION = '_mx_collect_adapter_version'
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
    if type(version) is not int or version != 2:
        return False
    for name, arguments in _HOOKS.items():
        functions = [node for node in tree.body
                     if isinstance(node, cst.FunctionDef) and node.name.value == name]
        if (len(functions) != 1 or functions[0].asynchronous is not None
                or not arguments <= {param.name.value for param in functions[0].params.params}):
            return False
    return True


def _requests_collection(module):
    for cls in module.classes.values():
        options = module.spec.get_spec(cls.fqname)
        if (options.get('serial_collect') is not None or options.get('parallel_collect_reuse')
                or options.get('serial_private_lookups') or options.get('parallel_collect_order') == 'domain'):
            return True
    return any(isinstance(node, cst.Call) and dotted_name(node.func) == '_mx_sys.collect_from_space'
               for node in walk(module.visitor.wrapper.module))


class CollectRuntime:
    def __init__(self, source):
        self.syntax = cst.parse_module(source)
        self.hook = _current(self.syntax)

    def require_current(self, modules, workers):
        if not self.hook and (workers != 1 or any(_requests_collection(module) for module in modules)):
            raise ValueError('Collection requires export protocol v2. Re-export the model with current modelx.')
