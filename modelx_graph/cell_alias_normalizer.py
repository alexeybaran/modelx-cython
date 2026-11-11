"""Conservative owner-local Cell call alias normalization (proposed).

Call after normalize_formula_function and before _BindAllCellCalls. Names in
owner_cell_names are metadata-authorized lexical Cell identities, not guesses.
Does not mutate its input. Rejected candidates retain their original source.
"""
from __future__ import annotations
import ast
import copy
from dataclasses import dataclass
from typing import Collection

@dataclass(frozen=True)
class AliasNormalization:
    function: ast.FunctionDef
    applied: dict[str, str]
    rejected: dict[str, str]

def normalize_cell_aliases(function: ast.FunctionDef,
                           owner_cell_names: Collection[str]) -> AliasNormalization:
    fn = copy.deepcopy(function)
    owner = set(owner_cell_names)
    applied, rejected = {}, {}
    params = {arg.arg for arg in (*fn.args.posonlyargs, *fn.args.args,
                                 *fn.args.kwonlyargs)}
    params.update(arg.arg for arg in (fn.args.vararg, fn.args.kwarg) if arg)
    nodes = list(ast.walk(fn))
    parents = {id(child): parent for parent in nodes for child in ast.iter_child_nodes(parent)}
    nested_scope = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.Lambda, ast.ClassDef)) and n is not fn for n in nodes)
    introspection = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id in {'locals', 'globals', 'vars', 'eval', 'exec',
                                          'setattr', 'delattr'} for n in nodes)
    # Catch bindings whose names are strings rather than ast.Name(Store).
    string_bindings = set()
    for n in nodes:
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            string_bindings.update(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            string_bindings.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            string_bindings.update(a.asname or a.name.split('.')[0] for a in n.names)
        elif isinstance(n, ast.MatchAs) and n.name:
            string_bindings.add(n.name)
        elif isinstance(n, ast.MatchStar) and n.name:
            string_bindings.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            string_bindings.add(n.rest)
    removed = set()
    for index, stmt in enumerate(fn.body):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)):
            continue
        alias = stmt.targets[0].id
        value = stmt.value
        target = value.id if isinstance(value, ast.Name) else (
            value.attr if isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name) and value.value.id == 'self' else None)
        if target not in owner:
            continue
        reason = None
        uses = [n for n in nodes if isinstance(n, ast.Name) and n.id == alias]
        stores = [n for n in uses if not isinstance(n.ctx, ast.Load)]
        loads = [n for n in uses if isinstance(n.ctx, ast.Load)]
        target_local = target in params or target in string_bindings or any(
            isinstance(n, ast.Name) and n.id == target
            and isinstance(n.ctx, (ast.Store, ast.Del)) for n in nodes)
        if nested_scope:
            reason = 'nested_scope'
        elif introspection:
            reason = 'observable_local_bindings_or_mutation'
        elif alias in params or alias in string_bindings or alias in owner:
            reason = 'alias_binding_conflict'
        elif len(stores) != 1 or stores[0] is not stmt.targets[0]:
            reason = 'alias_rebound_or_shadowed'
        elif target_local:
            reason = 'cell_name_shadowed'
        elif not loads:
            reason = 'unused_alias'
        elif any(not isinstance(parents.get(id(n)), ast.Call)
                 or parents[id(n)].func is not n for n in loads):
            reason = 'alias_escapes_call_position'
        elif any(any(n is child for child in ast.walk(earlier))
                 for n in loads for earlier in fn.body[:index+1]):
            reason = 'alias_used_before_binding'
        if reason:
            rejected[alias] = reason
            continue
        applied[alias] = target
        removed.add(id(stmt))
    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in applied:
                node.func = ast.copy_location(ast.Name(applied[node.func.id], ast.Load()), node.func)
            return node
    fn.body = [stmt for stmt in fn.body if id(stmt) not in removed]
    fn = Rewrite().visit(fn)
    return AliasNormalization(ast.fix_missing_locations(fn), applied, rejected)
