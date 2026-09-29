"""Import a bounded live-modelx tree without editing any lifelib formulas.

Data readers run at build time, then the generated package embeds their DataFrames.
One Data Space and one independent-policy dynamic Space are supported. Formulas
are source-preserved; ordinary Python objects need no speculative type inference.
"""
import ast
import hashlib
import inspect
import json
from enum import IntEnum
from pathlib import Path
import sys
import types
from .contracts import ContractError, cell
from .model_plan import derive_spec


def from_modelx(model, destination, *, projection='Projection', data_space='Data',
                point_table=None, steps=2048, replacements=None, contracts=None, signatures=None):
    """Derive a self-contained build spec; destination stores source evidence only.

    replacements is an explicit mapping of Cell names to replacement formula source.
    It is meant for reviewing author-side adaptations, never product detection.
    contracts optionally supplies ordinary annotation dictionaries by Cell name.
    """
    import pandas as pd
    dest=Path(destination).resolve();dest.mkdir(parents=True,exist_ok=True)
    import numpy as np
    space=getattr(model,projection)
    data=getattr(model,data_space,None)
    parameters=tuple(space.parameters or ())
    if space.spaces:
        raise ContractError('Live importer requires a flat Projection Space without children')
    # The first required Space parameter is the independent policy selector.
    # Additional positional parameters are supported when modelx supplies literal
    # defaults; model_plan already models them as ordinary per-item parameters.
    space_signature = inspect.signature(space.formula.func) if parameters else None
    if parameters:
        sig_parameters = tuple(space_signature.parameters.values())
        if tuple(p.name for p in sig_parameters) != parameters:
            raise ContractError('Projection Space formula signature does not match modelx parameters')
        if any(p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in sig_parameters):
            raise ContractError('Projection Space parameters must be ordinary positional parameters')
        if sig_parameters[0].default is not inspect.Parameter.empty:
            raise ContractError('The point selector must be a required Space parameter')
        if any(p.default is inspect.Parameter.empty for p in sig_parameters[1:]):
            raise ContractError('Additional Space parameters need literal defaults')
    tables={}
    if data is not None:
        for name,fn in data.cells.items():
            if fn.parameters:raise ContractError('Data readers must have no parameters')
            value=fn()
            if isinstance(value,pd.DataFrame):tables[name]=value
            # Existing Data-reader contract remains DataFrame-only. Materializing
            # dependent array readers and then replacing their source table could
            # leave stale arrays; do not silently broaden that contract.
    from .lookups import power as kernel_power
    modules={}; refs={}; object_references={}; intrinsic_references={}; enum_references={}
    for name,value in space.refs.items():
        if name in parameters or name in ('data','_model','_space','__builtins__'):continue
        if isinstance(value,types.ModuleType):modules[name]=value
        elif inspect.isclass(value) and issubclass(value, IntEnum):
            # model_plan has first-class IntEnum domain lowering for category/key
            # helpers. Preserve the symbolic class in the detached formula module.
            enum_references[name]=value
        elif name == 'power' and value is kernel_power:
            # Canonical real-valued power is a compiler intrinsic. Keep the live
            # Python binding for fallback evaluation; native lowering rewrites it
            # to the Cython helper already emitted by compiler.py.
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
        point=parameters[0]
        if point_table in object_references:point_table=object_references[point_table]
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
              'parameter_defaults':({p.name:p.default for p in list(space_signature.parameters.values())[1:]} if space_signature else {})}
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
        model_name=model.name,cython_objects=True)
    spec.model.update(static_projection=static,object_references=object_references)
    if signatures:
        from dataclasses import replace
        spec=replace(spec,cython_signatures=signatures)
    return spec
