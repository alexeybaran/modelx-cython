"""Import a bounded live-modelx tree without editing any lifelib formulas.

Data readers run at build time, then the generated package embeds their DataFrames.
One Data Space and one independent-policy dynamic Space are supported. Formulas
are source-preserved; ordinary Python objects need no speculative type inference.
"""
import ast
import hashlib
import inspect
import json
import symtable
from enum import IntEnum
from pathlib import Path
import sys
import types
from .contracts import ContractError, cell
from .model_plan import derive_spec


def _collect_sum_plan(space, point, point_table):
    """Select the numeric sum fast path or the general object-output traversal.

    Both paths traverse the full declared policy domain in its original order.
    The general path evaluates the source-preserved collect_outputs Cell once
    per policy and validates its result under the public collect contract.
    """
    domains = getattr(space._impl, 'collect_over_spec', None)
    if domains is None:
        return None
    if tuple(domains) != (point,) or 'collect_outputs' not in space.cells:
        raise ContractError('Frame collect requires the policy parameter and collect_outputs')
    domain = tuple(domains[point].value(space._impl))
    if domain != tuple(point_table.index):
        raise ContractError('Frame collect domain must match the policy-table index in order')
    source = space.cells['collect_outputs'].formula.source
    parsed = ast.parse(source).body
    if len(parsed) != 1:
        raise ContractError('Frame collect needs one collect_outputs formula')
    node = parsed[0]
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Lambda):
        result = node.value.body
    elif isinstance(node, ast.FunctionDef):
        body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                  and isinstance(node.body[0].value, ast.Constant)
                                  and isinstance(node.body[0].value.value, str)) else node.body
        returns = [stmt for stmt in body if isinstance(stmt, ast.Return)]
        if len(returns) != 1 or len(body) != 1:
            return {'over': point, 'object_outputs': 'collect_outputs'}
        result = returns[0].value
    else:
        return {'over': point, 'object_outputs': 'collect_outputs'}
    if not isinstance(result, ast.Dict) or len(result.keys) != 1:
        return {'over': point, 'object_outputs': 'collect_outputs'}
    mode = result.keys[0]
    values = result.values[0]
    if not isinstance(mode, ast.Constant) or mode.value != 'sum' or not isinstance(values, ast.Dict):
        return {'over': point, 'object_outputs': 'collect_outputs'}
    if not values.keys:
        return {'over': point, 'object_outputs': 'collect_outputs'}
    names = []
    for key, call in zip(values.keys, values.values):
        if (not isinstance(key, ast.Constant) or type(key.value) is not str
                or not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                or call.args or call.keywords or key.value != call.func.id
                or call.func.id not in space.cells):
            return {'over': point, 'object_outputs': 'collect_outputs'}
        names.append(key.value)
    if len(set(names)) != len(names):
        raise ContractError('Frame collect sum contains duplicate output names')
    return {'over': point, 'sum': names}


def from_modelx(model, destination, *, projection='Projection', data_space='Data',
                point_table=None, steps=2048, replacements=None, contracts=None, signatures=None,
                cython_objects=True, policy_records=None, row_references=None,
                parameter_initial_values=None):
    """Derive a self-contained build spec; destination stores source evidence only.

    replacements is an explicit mapping of Cell names to replacement formula source.
    It is meant for reviewing author-side adaptations, never product detection.
    contracts optionally supplies ordinary annotation dictionaries by Cell name.
    policy_records supplies prepared rows for a single independent-policy Space;
    row_references maps formula reference names to its object-valued columns.
    parameter_initial_values supplies internal initial values for required
    non-policy Space parameters; callers must still pass those parameters.
    Dynamic Space formulas are compiled from their original source with
    explicit references. They are not executed at build time.
    Connected Space preparation is not inferred by this flat importer.
    """
    import pandas as pd
    dest=Path(destination).resolve();dest.mkdir(parents=True,exist_ok=True)
    import numpy as np
    space=getattr(model,projection)
    data=getattr(model,data_space,None)
    row_references=dict(row_references or {})
    parameters=tuple(space.parameters or ())
    if space.spaces:
        raise ContractError('Live importer requires a flat Projection Space without children')
    # A declared collect domain identifies the independent policy selector even
    # when it is not the first Space parameter. The public signature stays in
    # author order; initial values only establish typed internal frame bindings.
    space_signature = inspect.signature(space.formula.func) if parameters else None
    domains = getattr(space._impl, 'collect_over_spec', None)
    if domains is not None and len(domains) != 1:
        raise ContractError('Frame collect requires one policy domain')
    point = next(iter(domains)) if domains else (parameters[0] if parameters else None)
    initial = dict(parameter_initial_values or {})
    if parameters:
        sig_parameters = tuple(space_signature.parameters.values())
        if tuple(p.name for p in sig_parameters) != parameters:
            raise ContractError('Projection Space formula signature does not match modelx parameters')
        if any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in sig_parameters):
            raise ContractError('Projection Space parameters must be ordinary positional parameters')
        if point not in parameters:
            raise ContractError('Collection selector is not a Space parameter')
        if space_signature.parameters[point].default is not inspect.Parameter.empty:
            raise ContractError('The point selector must be a required Space parameter')
        required = {p.name for p in sig_parameters
                    if p.name != point and p.default is inspect.Parameter.empty}
        if set(initial) != required:
            raise ContractError('Required non-policy Space parameters need initial values')
    tables={}
    data_program = None
    # A declared Data reference can itself be the source policy table. In that
    # layout the Data Cells often return position-indexed arrays, so preserve
    # both the references and those calculated arrays as separate bindings.
    reference_table = (data is not None and point_table is not None
                       and point_table in data.refs and point_table not in data.cells)
    data_references = {}
    if data is not None:
        if reference_table:
            for name, value in data.refs.items():
                if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
                    if name in data.cells:
                        raise ContractError('Data reference/Cell name collision: ' + name)
                    data_references[name] = value
                    tables[name] = value
        for name,fn in data.cells.items():
            if fn.parameters:raise ContractError('Data readers must have no parameters')
            value=fn()
            if isinstance(value,pd.DataFrame) or (reference_table and isinstance(value,(pd.Series,np.ndarray))):
                tables[name]=value
            # Source-backed Data references are refreshed with their dependent
            # arrays as one input revision below. The older DataFrame-Cell route
            # keeps its existing binding contract.
    if policy_records is not None:
        if not isinstance(policy_records, pd.DataFrame):
            raise ContractError('Prepared policy records must be a DataFrame')
        if point_table is None:
            point_table='mxcf_policy_records'
        if point_table in tables:
            raise ContractError('Prepared policy records conflict with a Data table: '+point_table)
        tables[point_table]=policy_records
    from .lookups import power as kernel_power
    modules={}; refs={}; object_references={}; intrinsic_references={}; enum_references={}
    cell_loads = {node.id for fn in space.cells.values()
                  for node in ast.walk(ast.parse(fn.formula.source))
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    for name,value in space.refs.items():
        if name in parameters or name in ('data','_model','_space','__builtins__'):continue
        if name in row_references:continue
        if name not in cell_loads:continue
        if isinstance(value,types.ModuleType):modules[name]=value
        elif inspect.isclass(value) and issubclass(value, IntEnum):
            # model_plan has first-class IntEnum domain lowering for category/key
            # helpers. Preserve the symbolic class in the detached formula module.
            enum_references[name]=value
        elif name == 'power' and (value is kernel_power or
                (callable(value) and getattr(value, '__modelx_cython_intrinsic__', None)
                 == 'real_power_v1')):
            # A helper author can explicitly assert the canonical real-power
            # contract. Keep the actual Python binding for fallback execution;
            # eligible native formulas use the generated checked intrinsic.
            intrinsic_references[name]=value
        elif type(value) in (int,float,bool,str,type(None)):
            refs[name]=value
        elif isinstance(value,(pd.DataFrame,pd.Series,np.ndarray)):
            table='reference_'+name
            if table in tables:raise ContractError('Conflicting detached Data name: '+table)
            tables[table]=value;object_references[name]=table
        else:raise ContractError('Unsupported detached reference: '+name)
    static=not parameters
    if static:
        point='mxcf_scope_id'; point_table='mxcf_static_scope'
        if point_table in tables:raise ContractError('Reserved static scope Data name is already used')
        tables[point_table]=pd.DataFrame(index=pd.Index([0],name=point))
    else:
        if point_table in object_references:point_table=object_references[point_table]
        if reference_table:
            if not isinstance(tables.get(point_table), pd.DataFrame):
                raise ContractError('Declared source policy reference must be a DataFrame')
            if domains is None or tuple(domains) != (point,):
                raise ContractError('Position-indexed Data references require a declared collect domain')
            domain_name = getattr(domains[point], 'ref_name', None)
            if domain_name not in space.cells:
                raise ContractError('Collection domain must name a source-preserved Cell')
            from .data_program import prepare_data
            data_modules = {name: value.__name__ for name, value in data.refs.items()
                            if isinstance(value, types.ModuleType)}
            data_program = dict(references=sorted(data_references), modules=data_modules,
                                cells={name: fn.formula.source for name, fn in data.cells.items()},
                                source_policy_table=point_table, domain_name=domain_name,
                                domain_source=space.cells[domain_name].formula.source,
                                policy_table='mxcf_policy_domain')
            if data_program['policy_table'] in tables:
                raise ContractError('Reserved policy-domain Data name is already used')
            derived, domain = prepare_data(data_program, data_references)
            for name, value in derived.items():
                actual = tables.get(name)
                if actual is None or type(actual) is not type(value):
                    raise ContractError('Data source round trip changed object kind: ' + name)
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    same = actual.equals(value)
                else:
                    try: np.testing.assert_array_equal(actual, value); same = True
                    except AssertionError: same = False
                if not same:
                    raise ContractError('Data source round trip changed values: ' + name)
            declared_domain = tuple(domains[point].value(space._impl))
            if domain != declared_domain:
                raise ContractError('Data source round trip changed collection domain')
            tables[data_program['policy_table']] = pd.DataFrame(index=pd.Index(domain, name=point))
            point_table = data_program['policy_table']
        if point_table is None:
            candidates=[n for n,t in tables.items() if isinstance(t,pd.DataFrame)
                        and list(t.index.names)==[point]]
            if len(candidates)!=1:raise ContractError('Declare point_table: policy index is ambiguous')
            point_table=candidates[0]
    replacements=dict(replacements or {}); contracts=dict(contracts or {})
    if (replacements.keys()|contracts.keys())-space.cells.keys():
        raise ContractError('Unknown formula replacement/contract')
    chunks=['# Derived from modelx formulas. Original model files are unchanged.']
    original={}; normalized=[]
    for name,fn in space.cells.items():
        if getattr(fn, 'is_cached', True) is False:
            raise ContractError('Frame frontend does not yet preserve uncached Cell behavior: '+name)
        text=fn.formula.source
        original[name]=hashlib.sha256(text.encode()).hexdigest()
        chosen=replacements.get(name,text)
        node=ast.parse(chosen).body[0]
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Lambda):
            lam=node.value
            node=ast.FunctionDef(name=name,args=lam.args,body=[ast.Return(lam.body)],decorator_list=[])
            chosen=ast.unparse(ast.fix_missing_locations(node));normalized.append(name)
        if not isinstance(node,ast.FunctionDef) or node.name!=name:
            raise ContractError('Formula must be a named definition or a top-level lambda: '+name)
        chunks.append(chosen)
    path=dest/'formulas.py';path.write_text('\n\n'.join(chunks)+'\n', encoding='utf-8')
    module_name='_mxcf_formula_'+hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    module=types.ModuleType(module_name);module.__file__=str(path)
    module.__dict__.update(modules)
    module.__dict__.update(intrinsic_references)
    module.__dict__.update(enum_references)
    from . import lookup
    module.lookup = lookup
    # Do not bake point selectors, reference defaults or modelx objects into globals.
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),module.__dict__)
    sys.modules[module_name]=module
    for name in space.cells:
        fn=module.__dict__[name]; cell(fn)
        fn.__mxcf_original_source__=space.cells[name].formula.source
        fn.__mxcf_syntax_normalization__='lambda_cell_to_def' if name in normalized else None
        if name in contracts:fn.__annotations__=dict(contracts[name])
    if static:
        scope={};exec('def space_formula('+point+'):\n    pass',scope)
        space_formula=scope['space_formula']
    else:
        # derive_spec only inspects this signature. Passing the live modelx Space
        # formula preserves declared defaults without evaluating model logic.
        space_formula=space.formula.func
    evidence={'model':model.name,'formula_sha256':original,'replaced_cells':sorted(replacements),
              'model_source_unchanged':True,'point_table':point_table,'data_tables':sorted(tables),
              'static_projection':static,'object_references':object_references,
              'intrinsic_references':sorted(intrinsic_references),
              'enum_references':sorted(enum_references),
              'lambda_cells_normalized':normalized,
              'space_parameters':list(parameters),
              'row_references':row_references,
              'parameter_defaults':({p.name:(initial[p.name] if p.name in initial else p.default)
                                      for p in space_signature.parameters.values() if p.name != point}
                                     if space_signature else {}),
              'required_parameters':sorted(initial)}
    if data_program is not None:
        evidence['data_reference_tables'] = sorted(data_references)
        evidence['policy_domain_source'] = data_program['domain_name']
    (dest/'frontend.json').write_text(json.dumps(evidence,indent=2), encoding='utf-8')
    def effective_allow_none(obj):
        while obj is not None:
            value=getattr(obj,'allow_none',None)
            if value is not None:return bool(value)
            obj=getattr(obj,'parent',None)
        return False
    spec = derive_spec(module,tables,references=refs,space_formula=space_formula,
        outputs={},length='0',steps=steps,cell_outputs=tuple(space.cells),
        data_space=data_space,projection=projection,point_table=point_table,
        allow_none={n:effective_allow_none(c) for n,c in space.cells.items()},
        model_name=model.name,cython_objects=cython_objects,
        row_references=row_references,point_parameter=point,
        parameter_initial_values=initial)
    spec.model.update(static_projection=static,object_references=object_references,
                      module_references={name: value.__name__ for name, value in modules.items()})
    if space.formula is not None:
        graph = getattr(model, 'source_graph', None)
        name = model.name + '.' + projection
        source = (graph['spaces'][name]['formula'] if graph is not None else
                  getattr(space.formula, 'source', None))
        if not isinstance(source, str):
            raise ContractError('Space formula source is required for Cython preparation')
        scopes = symtable.symtable(source, '<space formula>', 'exec').get_children()
        if len(scopes) != 1 or scopes[0].get_type() != 'function':
            raise ContractError('Space formula must define one function')
        scope = scopes[0]
        used_refs = sorted(name for name in space.refs
                           if name in scope.get_identifiers()
                           and scope.lookup(name).is_global()
                           and scope.lookup(name).is_referenced())
        spec.model['space_formula'] = {
            'source': source, 'sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
            'references': used_refs}
    if data_program is not None:
        spec.model['data_program'] = data_program
        spec.model['data_reference_names'] = sorted(data_references)
    collect = _collect_sum_plan(space, point, tables[point_table]) if not static else None
    if collect is not None:
        from dataclasses import replace
        spec.model['collect'] = collect
        if 'sum' in collect:
            spec = replace(spec, scalar_outputs=(*spec.scalar_outputs, *collect['sum']))
    if signatures:
        from dataclasses import replace
        spec=replace(spec,cython_signatures=signatures)
    return spec
