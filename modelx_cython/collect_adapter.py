"""Versioned optional collection capability; ordinary exports stay usable.

The exporter owns the runtime. The known pre-versioned implementation may
be upgraded with the original v1 entry points bundled with the compiler.
"""
import hashlib

import libcst as cst

from .helpers import read_resource
from .scopes import Names
from .syntax import literal_value, semantic_key, statements


_VERSION = '_mx_collect_adapter_version'
_WORKERS = '_mx_parallel_workers'
_LEGACY = 'b463114e5bab0f7733e2bb8b054f0b596ea6859532ffc16a73d064febfc2d598'
_ARGUMENTS = {
    'collect_from_space': ('evaluate_serial', 'reuse_in_parallel', 'private_ordered'),
    '_mx_collect_parallel': ('private', 'private_ordered'),
    '_mx_collect_chunk': ('private',),
}


def _assignments(tree, name):
    return [node for node in statements(tree) if isinstance(node, cst.Assign)
            and any(isinstance(target.target, cst.Name) and target.target.value == name
                    for target in node.targets)]


def _version(tree):
    if any(isinstance(node, (cst.AnnAssign, cst.AugAssign))
           and isinstance(node.target, cst.Name) and node.target.value == _VERSION
           for node in statements(tree)):
        return None
    nodes = _assignments(tree, _VERSION)
    if not nodes:
        return 0
    if len(nodes) == 1:
        try:
            value = literal_value(nodes[0].value)
            if type(value) is int:
                return value
        except (ValueError, TypeError):
            pass
    return None


def _functions(tree):
    names = (*_ARGUMENTS, '_mx_collect_native_coords')
    return {name: [node for node in tree.body if isinstance(node, cst.FunctionDef)
                   and node.asynchronous is None and node.name.value == name] for name in names}


def _legacy(tree):
    # The callback replaces evaluation as well as dispatch. Fingerprint the
    # complete collection protocol, independent of function order/formatting.
    functions = sorted((node for node in tree.body
        if isinstance(node, cst.FunctionDef)
        and (node.name.value == 'collect_from_space' or node.name.value.startswith('_mx_collect_'))),
        key=lambda node: node.name.value)
    return hashlib.sha256(repr(semantic_key(functions)).encode()).hexdigest() == _LEGACY


def _native(tree, version=1):
    functions = _functions(tree)
    return (_version(tree) == version and all(len(nodes) == 1 for nodes in functions.values())
            and all(all(name in [arg.name.value for arg in functions[function][0].params.params]
                        for name in arguments) for function, arguments in _ARGUMENTS.items()))


class CollectRuntime:
    def __init__(self, source):
        self.syntax = cst.parse_module(source)
        self.template = None
        self.hook = _native(self.syntax, 2)
        self.supported = self.hook or _native(self.syntax)
        if self.supported or _version(self.syntax) != 0:
            return
        legacy = (not Names(self.syntax).names.intersection(
            {_VERSION, '_mx_collect_native_coords'}) and _legacy(self.syntax))
        if legacy:
            source = read_resource(__package__, '_collect_v1.py').decode('utf-8')
            self.template = cst.parse_module(source)
            self.supported = True

    def render(self, workers=1, native=False):
        if self.hook:
            return self.syntax.code
        if native and not self.supported:
            raise ValueError('unsupported native collection adapter')
        settings = _assignments(self.syntax, _WORKERS)
        if workers != 1 and (len(settings) != 1 or len(settings[0].targets) != 1):
            raise ValueError('parallel_workers requires a collection-enabled modelx export')
        replacements = {}
        if native and self.template is not None:
            replacements = {node.name.value: node for node in self.template.body
                            if isinstance(node, cst.FunctionDef) and node.name.value in _ARGUMENTS}
        body = []
        for node in self.syntax.body:
            if isinstance(node, cst.SimpleStatementLine) and len(node.body) == 1:
                item = node.body[0]
                if (workers != 1 and isinstance(item, cst.Assign) and len(item.targets) == 1
                        and isinstance(item.targets[0].target, cst.Name)
                        and item.targets[0].target.value == _WORKERS):
                    node = node.with_changes(body=(item.with_changes(value=cst.Integer(str(workers))),))
            if isinstance(node, cst.FunctionDef) and node.name.value in replacements:
                if node.name.value == 'collect_from_space':
                    body.append(cst.parse_statement(_VERSION + ' = 1\n'))
                    body.extend(item for item in self.template.body if isinstance(item, cst.FunctionDef)
                                and item.name.value == '_mx_collect_native_coords')
                node = replacements[node.name.value]
            body.append(node)
        return self.syntax.with_changes(body=body).code
