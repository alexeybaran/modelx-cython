"""Build-time formula provenance and explicit Cython eligibility evidence.

These are syntax diagnostics, NOT a Python-free proof or a pandas interpreter.
The eligibility predicate is shared with the emitter so fallback reasons cannot
silently drift from the decision that selected a detached Python calculator.
"""
import ast
import hashlib
import inspect
from modelx_cython.formula_emitter import formula_source, formula_location
import textwrap


def cython_fallback_reasons(fn, cell_names, *, forced=False):
    """Explain only the bounded GIL-Cython boundary, not scalar/native typing."""
    node=ast.parse(textwrap.dedent(formula_source(fn))).body[0]
    reasons=[]
    if forced:reasons.append('explicit_python_override')
    kinds=((ast.Lambda,'lambda'),(ast.Yield,'yield'),(ast.YieldFrom,'yield_from'),
           (ast.AsyncFunctionDef,'async_function'),(ast.Global,'global_mutation'),
           (ast.Nonlocal,'nonlocal_mutation'))
    for kind,code in kinds:
        if any(isinstance(n,kind) for n in ast.walk(node)):reasons.append(code)
    if any(isinstance(n,ast.FunctionDef) and n is not node for n in ast.walk(node)):
        reasons.append('nested_function')
    parents={id(child):parent for parent in ast.walk(node) for child in ast.iter_child_nodes(parent)}
    reducers={'sum','any','all','min','max'}
    for g in (n for n in ast.walk(node) if isinstance(n,ast.GeneratorExp)):
        parent=parents[id(g)]
        if not (isinstance(parent,ast.Call) and isinstance(parent.func,ast.Name)
                and parent.func.id in reducers and parent.func.id not in fn.__code__.co_varnames
                and parent.func.id not in fn.__globals__):
            reasons.append('deferred_or_unrecognized_generator');break
    calls={id(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)}
    # Local variables may share names with unrelated Cells. Python's lexical
    # binding information distinguishes them without evaluating formulas.
    global_cells = set(cell_names) - set(fn.__code__.co_varnames) - set(fn.__code__.co_cellvars)
    if any(isinstance(n,ast.Name) and n.id in global_cells and id(n) not in calls for n in ast.walk(node)):
        reasons.append('cell_as_value')
    if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in global_cells
           and (any(isinstance(a,ast.Starred) for a in n.args) or any(k.arg is None for k in n.keywords))
           for n in ast.walk(node)):
        reasons.append('expanded_cell_arguments')
    return reasons


def record_formula_coverage(manifest,spec,forced_python=()):
    """Attach per-formula evidence without changing the generated hot path."""
    counts={};forced=set(forced_python)
    for name,row in manifest['cells'].items():
        fn=spec.cells[name];source=textwrap.dedent(formula_source(fn));tree=ast.parse(source)
        source_file, source_line = formula_location(fn)
        calculator=row.get('calculator','native')
        backend={'native':'cython_nogil','cython_scalar':'cython_gil',
                 'cython_object':'cython_gil','python':'detached_python'}[calculator]
        counts[backend]=counts.get(backend,0)+1
        reasons=cython_fallback_reasons(fn,manifest['cells'],forced=name in forced) if backend=='detached_python' else []
        if backend=='detached_python' and not reasons:
            reasons=['scalar_contract_or_unsupported_expression']
        modules=sorted({n.func.value.id for n in ast.walk(tree)
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
            and inspect.ismodule(fn.__globals__.get(n.func.value.id))})
        row['coverage']={'body_backend':backend,'detached_python':backend=='detached_python',
            'fallback_reasons':reasons,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'source_file':source_file,'source_line':source_line,
            'module_call_names':modules,
            'contains_attribute_calls':any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) for n in ast.walk(tree)),
            'contains_comprehension':any(isinstance(n,(ast.ListComp,ast.DictComp,ast.SetComp,ast.GeneratorExp)) for n in ast.walk(tree)),
            'python_free_proven':False,
            'original_source_sha256':hashlib.sha256(getattr(fn,'__mxcf_original_source__',source).encode()).hexdigest(),
            'syntax_normalization':getattr(fn,'__mxcf_syntax_normalization__',None)}
    # Only claim a Python-free closure when every transitive Cell body is emitted
    # into the checked nogil backend. A Cython scalar return type is insufficient.
    for name,row in manifest['cells'].items():
        pending=[name];seen=set();proven=True
        while pending:
            item=pending.pop()
            if item in seen:continue
            seen.add(item);dep=manifest['cells'].get(item)
            if dep is None or dep['coverage']['body_backend']!='cython_nogil':
                proven=False;break
            pending.extend(dep.get('dependencies',()))
        row['coverage']['python_free_proven']=proven
    manifest['formula_coverage']={'schema':'mxcf.formula_coverage.v1','counts':counts,
        'gil_cython_is_not_python_free':True,'runtime_miss_counts':'Collected separately; zero is not proof of native coverage.'}
