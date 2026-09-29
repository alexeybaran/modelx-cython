"""Re-evaluate source-preserved Data Cells after a Data reference is replaced.

The formulas execute as Python functions at the input-assignment boundary. This
does not interpret Projection formulas or put Data preparation inside the policy
loop. All results are prepared before a new input revision is published.
"""
import ast
import importlib

from .contracts import ContractError


def _function(source, name, namespace):
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != name:
        raise ContractError('Data formula must be one named function: ' + name)
    exec(compile(tree, '<compiled Data formula: ' + name + '>', 'exec'), namespace)
    return namespace[name]


def prepare_data(program, references):
    """Return all derived Data objects and the declared collection domain.

    A fresh namespace and memo are used per input revision. Cycles fail rather
    than publishing an incomplete binding; dependent Data Cells share results.
    """
    import numpy as np
    import pandas as pd

    names = set(program['references'])
    if set(references) != names:
        raise ContractError('Data reference names differ from the compiled source')
    namespace = dict(references)
    for alias, module in program['modules'].items():
        if alias in namespace:
            raise ContractError('Data module/reference name collision: ' + alias)
        namespace[alias] = importlib.import_module(module)
    sources = program['cells']
    if (set(sources) & set(namespace)) or len(sources) != len(set(sources)):
        raise ContractError('Data Cell/reference name collision')
    bodies = {name: _function(source, name, namespace) for name, source in sources.items()}
    completed, active = {}, set()

    def evaluate(name):
        if name in completed:
            return completed[name]
        if name in active:
            raise ContractError('Recursive Data Cell: ' + name)
        active.add(name)
        try:
            value = bodies[name]()
            if not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
                raise ContractError('Data Cell must return a DataFrame, Series or ndarray: ' + name)
            completed[name] = value
            return value
        finally:
            active.remove(name)

    for name in sources:
        namespace[name] = (lambda cell=name: evaluate(cell))
    for name in sources:
        evaluate(name)

    class _DataReceiver:
        def __getattr__(self, name):
            if name in completed:
                return lambda: completed[name]
            if name in references:
                return references[name]
            raise AttributeError(name)

    domain_name = program['domain_name']
    domain_source = program['domain_source']
    domain_namespace = {'data': _DataReceiver()}
    domain = tuple(_function(domain_source, domain_name, domain_namespace)())
    if len(domain) != len(set(domain)):
        raise ContractError('Collection policy domain has duplicate keys')
    if len(domain) != len(references[program['source_policy_table']]):
        raise ContractError('Collection policy domain and source table lengths differ')
    return completed, domain
