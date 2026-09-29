"""Read explicit scalar signatures from a standard modelx-cython exported source.

This is an optional, build-time bridge, not a new type tracer. Observed signatures
are contracts, not proof about unseen Python inputs. Unrecognised types stay
objects, and numeric entry arguments are checked before conversion.
"""
import ast
from pathlib import Path
from .contracts import ContractError

_TYPES = {'int':'int','long':'int','longlong':'int','double':'float','float':'float',
          'bint':'bool','bool':'bool','str':'str','object':'object'}

def from_exported_cython(path, *, space='Projection'):
    root=ast.parse(Path(path).read_text())
    classes=[n for n in root.body if isinstance(n,ast.ClassDef) and n.name=='_c_'+space]
    if len(classes)!=1:raise ContractError('Expected one exported Space class: '+space)
    def kind(node):
        name=node.id if isinstance(node,ast.Name) else node.attr if isinstance(node,ast.Attribute) else ''
        return _TYPES.get(name,'object')
    result={}
    for fn in classes[0].body:
        if isinstance(fn,ast.FunctionDef) and fn.name.startswith('_f_'):
            result[fn.name[3:]]={'return':kind(fn.returns),
                'parameters':{a.arg:kind(a.annotation) for a in fn.args.posonlyargs+fn.args.args if a.arg!='self'}}
    return result
