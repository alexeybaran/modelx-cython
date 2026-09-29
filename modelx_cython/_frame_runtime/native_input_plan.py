"""Expand declared numeric row references into native physical inputs.

The source formulas and public row references remain unchanged. Cython still
compiles each formula; this plan supplies a typed binding for selected reads.
"""
from copy import deepcopy
from dataclasses import replace
import types

from .contracts import ContractError
from .lookups import IndexedFloat1


_KINDS = {'float': float, 'float64_1d': IndexedFloat1}
_ANNOTATIONS = {'float': float, 'int': int, 'bool': bool}


def expand(spec, signatures, direct_inputs):
    """Return an experimental spec with one native binding per declared read."""
    mappings = spec.typed_mapping_inputs or {}
    direct_inputs = dict(direct_inputs or {})
    declarations = [(name, key, kind)
                    for name, fields in sorted(mappings.items())
                    for key, kind in sorted(fields.items())]
    declarations += [(name, None, kind) for name, kind in sorted(direct_inputs.items())]
    if not declarations:
        raise ContractError('Native input expansion needs typed declarations')
    model = deepcopy(spec.model)
    if model is None:
        raise ContractError('Native row references need a model binding plan')
    inputs = dict(spec.inputs)
    frontend = deepcopy(spec.frontend or {})
    fields = frontend.setdefault('native_mapping_fields', {})
    direct = frontend.setdefault('native_direct_inputs', {})
    bindings = model['bindings']
    for owner, key, kind in declarations:
        if kind not in _KINDS:
            raise ContractError('Unsupported native row-reference type: ' + kind)
        parent = bindings.get(owner)
        if parent is None or parent['kind'] != 'row_reference':
            raise ContractError('Native input must derive from a row reference: ' + owner)
        if key is None and owner in mappings:
            raise ContractError('A row reference cannot be both direct and mapped: ' + owner)
        physical = 'mxinput_' + owner + ('_' + key if key is not None else '')
        if not physical.isidentifier() or physical in inputs or physical in spec.cells:
            raise ContractError('Native input name collides: ' + physical)
        inputs[physical] = _KINDS[kind]
        bindings[physical] = {
            'kind': 'row_reference_numeric', 'table': parent['table'],
            'column': parent['column'], 'field': key, 'value_kind': kind,
        }
        if key is None:
            direct[owner] = physical
        else:
            fields.setdefault(owner, {})[key] = physical
    clone = types.ModuleType(spec.module.__name__)
    clone.__dict__.update(vars(spec.module))
    for name, hint in signatures.items():
        if name not in spec.cells or hint['return'] not in _ANNOTATIONS:
            continue
        if any(kind not in _ANNOTATIONS for kind in hint['parameters'].values()):
            continue
        fn = spec.cells[name]
        bound = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__,
                                   fn.__defaults__, fn.__closure__)
        bound.__dict__.update(fn.__dict__)
        bound.__annotations__ = {
            **{key: _ANNOTATIONS[kind]
               for key, kind in hint['parameters'].items()},
            'return': _ANNOTATIONS[hint['return']],
        }
        bound.__module__ = fn.__module__
        setattr(clone, name, bound)
    model['native_row_references'] = {
        name: binding for name, binding in bindings.items()
        if binding['kind'] == 'row_reference_numeric'
    }
    return replace(spec, module=clone, inputs=inputs, frontend=frontend,
                   model=model, typed_mapping_inputs=None)
