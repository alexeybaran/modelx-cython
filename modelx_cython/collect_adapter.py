"""Versioned optional collection capability; ordinary exports stay usable.

The exporter owns the runtime. Only the known pre-versioned implementation
may be upgraded, by copying its three generic functions from that exporter.
"""
import ast
import hashlib

import libcst as cst

from .helpers import read_resource
from .scopes import Names


_VERSION = '_mx_collect_adapter_version'
_WORKERS = '_mx_parallel_workers'
_LEGACY = '2f048f81848e8b43db550d4eb4b747e805737ce15a18cb215d2d92a796bdb4e7'
_ARGUMENTS = {
    'collect_from_space': ('evaluate_serial', 'reuse_in_parallel', 'private_ordered'),
    '_mx_collect_parallel': ('private', 'private_ordered'),
    '_mx_collect_chunk': ('private',),
}


def _normalized(node):
    if isinstance(node, getattr(ast, 'Index', ())):
        return _normalized(node.value)
    # Empty type_params appeared in Python 3.12. Ignore that optional field
    # so the same legacy export is recognized on every supported Python.
    if isinstance(node, ast.AST):
        return type(node).__name__, tuple((key, _normalized(value))
            for key, value in ast.iter_fields(node) if key != 'type_params')
    return tuple(map(_normalized, node)) if isinstance(node, list) else node


def _assignments(tree, name):
    return [node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]


def _version(tree):
    if any(isinstance(node, (ast.AnnAssign, ast.AugAssign))
           and isinstance(node.target, ast.Name) and node.target.id == _VERSION
           for node in tree.body):
        return None
    nodes = _assignments(tree, _VERSION)
    if not nodes:
        return 0
    if len(nodes) == 1 and isinstance(nodes[0].value, ast.Constant):
        value = nodes[0].value.value
        if type(value) is int:
            return value
    return None


def _functions(tree):
    names = (*_ARGUMENTS, '_mx_collect_native_coords')
    return {name: [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == name] for name in names}


def _legacy(tree):
    # The callback replaces evaluation as well as dispatch. Fingerprint the
    # complete collection protocol, independent of function order/formatting.
    functions = sorted((node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (node.name == 'collect_from_space' or node.name.startswith('_mx_collect_'))),
        key=lambda node: node.name)
    return hashlib.sha256(repr(_normalized(functions)).encode()).hexdigest() == _LEGACY


def _native(tree):
    functions = _functions(tree)
    return (_version(tree) == 1 and all(len(nodes) == 1 for nodes in functions.values())
            and all(all(name in [arg.arg for arg in functions[function][0].args.args]
                        for name in arguments) for function, arguments in _ARGUMENTS.items()))


class CollectRuntime:
    def __init__(self, source):
        self.syntax = cst.parse_module(source)
        self.tree = ast.parse(source)
        self.template = None
        self.supported = _native(self.tree)
        if self.supported or _version(self.tree) != 0:
            return
        legacy = (not Names(self.syntax).names.intersection(
            {_VERSION, '_mx_collect_native_coords'}) and _legacy(self.tree))
        if legacy:
            try:
                source = read_resource('modelx.export', '_mx_sys.py').decode('utf-8')
            except (ImportError, FileNotFoundError):
                return
            if _native(ast.parse(source)):
                self.template = cst.parse_module(source)
                self.supported = True

    def render(self, workers=1, native=False):
        if native and not self.supported:
            raise ValueError('unsupported native collection adapter')
        settings = _assignments(self.tree, _WORKERS)
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
