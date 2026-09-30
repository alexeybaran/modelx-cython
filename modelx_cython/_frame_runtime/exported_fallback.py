"""Recover a conservative formula interface from an older modelx export.

Exports predating ``_mx_formula_sources.json`` still contain Cython-compilable
formula methods. This adapter reverses only the exporter's mechanical ``self``
qualification. It never evaluates a formula or guesses a result from samples.
The resulting source provenance is marked as reconstructed, since the exact
author spelling cannot be recovered from the generated Python file.
"""

import ast
import hashlib
import inspect
import sys
import textwrap

from .contracts import ContractError


class _UnbindFormula(ast.NodeTransformer):
    def __init__(self):
        self.references = set()

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            if node.attr.startswith('_'):
                raise ContractError('Cannot recover an exported private formula access')
            self.references.add(node.attr)
            return ast.copy_location(ast.Name(node.attr, node.ctx), node)
        return self.generic_visit(node)


def _method_source(method, cell_name):
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise ContractError('Cannot recover exported formula source: ' + cell_name) from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ContractError('Expected one exported formula method: ' + cell_name)
    node = tree.body[0]
    if (not node.args.args or node.args.args[0].arg != 'self'
            or node.args.posonlyargs or node.decorator_list):
        raise ContractError('Unsupported exported formula method: ' + cell_name)
    node.args.args.pop(0)
    node.name = cell_name
    unbinder = _UnbindFormula()
    node = ast.fix_missing_locations(unbinder.visit(node))
    return ast.unparse(node) + '\n', unbinder.references


def _assigned_references(space):
    """Read names the exporter explicitly assigns as Space references."""
    method = getattr(type(space), '_mx_assign_refs', None)
    if method is None:
        return set()
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise ContractError('Cannot recover exported Space references') from exc
    return {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name) and node.value.id == 'self'
            and not node.attr.startswith('_')}


def recover_source_manifest(exported):
    """Return an in-memory source manifest for an older, metadata-free export."""
    model_name = getattr(exported, '_name', None)
    if not isinstance(model_name, str) or not model_name.isidentifier():
        raise ContractError('Exported model has no usable identity')
    spaces = {}

    def visit(space, path):
        module = sys.modules[type(space).__module__]
        cell_names = getattr(module, '_v_cells_names_' + space._name, None)
        if not isinstance(cell_names, list):
            raise ContractError('Exported Space has no Cell inventory: ' + path)
        parameters = getattr(module, '_v_space_params_' + space._name, ())
        if not isinstance(parameters, (list, tuple)):
            raise ContractError('Exported Space has no parameter inventory: ' + path)
        cells = {}
        reads = set()
        for name in cell_names:
            if not isinstance(name, str) or not name.isidentifier():
                raise ContractError('Invalid exported Cell name: ' + path)
            body = getattr(type(space), '_f_' + name, None)
            cached = body is not None
            if body is None:
                body = getattr(type(space), name, None)
            if body is None:
                raise ContractError('Missing exported formula method: ' + path + '.' + name)
            source, uses = _method_source(body, name)
            reads.update(uses)
            cells[name] = {'source': source,
                           'sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
                           'cached': cached,
                           'allow_none': bool(getattr(space, '_allow_none', False))}
        references = sorted((_assigned_references(space) | reads)
                            - set(cell_names) - set(parameters))
        formula = None
        if parameters:
            generated = getattr(type(space), '_mx_space_formula', None)
            if generated is not None:
                formula, formula_reads = _method_source(generated, '_formula')
                references = sorted(set(references) | formula_reads - set(parameters))
            else:
                signature = inspect.signature(space.__call__)
                formula = ('def _formula' + str(signature)
                           + ":\n    return {'refs': {}}\n")
        targets = {}
        for name in references:
            try:
                value = getattr(space, name)
            except AttributeError as exc:
                raise ContractError('Unbound exported reference: ' + path + '.' + name) from exc
            if hasattr(value, '_mx_spaces') and hasattr(value, '_name'):
                parts = []
                current = value
                while hasattr(current, '_parent') and current is not exported:
                    parts.append(current._name)
                    current = current._parent
                if current is exported:
                    targets[name] = '.'.join([model_name, *reversed(parts)])
        fullname = model_name + '.' + path
        spaces[fullname] = {'formula': formula, 'parameters': list(parameters),
                            'references': references,
                            'reference_targets': targets, 'cells': cells,
                            'collect': None}
        for child_name, child in getattr(space, '_mx_spaces', {}).items():
            visit(child, path + '.' + child_name)

    for name, space in getattr(exported, '_mx_spaces', {}).items():
        visit(space, name)
    if not spaces:
        raise ContractError('Exported model has no Spaces')
    return {'schema': 'modelx.formula_sources.v2', 'model': model_name,
            'source_provenance': 'reconstructed_export_methods', 'spaces': spaces}
