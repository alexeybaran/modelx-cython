"""One source handoff with explicit receiver bindings, never formula evaluation."""
import ast
import copy
import inspect
import types


class FormulaEmitter:
    """Select a receiver representation without changing formula operations."""

    @staticmethod
    def ordinary(cell):
        return copy.deepcopy(cell.formula_node)

    @staticmethod
    def native(cell, argument_types, result_type, *, source_node=None, reference_aliases=None):
        from ._frame_runtime.exported_fallback import _UnbindFormula
        node = copy.deepcopy(source_node) if source_node is not None else FormulaEmitter.ordinary(cell)
        aliases = reference_aliases or {}
        class BindReceiver(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == 'self' and node.attr in aliases:
                    return ast.copy_location(ast.Attribute(node.value, aliases[node.attr], node.ctx), node)
                return self.generic_visit(node)
        node = BindReceiver().visit(node)
        node.name = cell.name
        if not node.args.args or node.args.args[0].arg != 'self':
            raise ValueError('Exported formula needs its original receiver')
        node.args.args.pop(0)
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = ast.Name(getattr(argument_types[arg.arg], '__name__', 'object'), ast.Load())
        node.returns = ast.Name(getattr(result_type, '__name__', 'object'), ast.Load())
        return ast.fix_missing_locations(_UnbindFormula().visit(node))


class SourceFormula:
    """Build-time function description; callable only for inspect.signature.

    Compile lexical code for existing global-name discovery but never execute it.
    Cython compiles the source body in the generated artifact.
    """
    __kernel_cell__ = True
    __closure__ = None

    def __init__(self, node, arguments, result, *, cached=True, filename,
                 module_name, globals_=None, signature=None):
        self.node = copy.deepcopy(node)
        self.source = ast.unparse(node) + '\n'
        self.filename = filename
        self.__name__ = self.__qualname__ = node.name
        self.__module__ = module_name
        self.__globals__ = {} if globals_ is None else globals_
        self.__kernel_cached__ = cached
        self.__annotations__ = dict(arguments, **{'return': result})
        if signature is None:
            defaults = dict(zip((a.arg for a in node.args.args[-len(node.args.defaults):]),
                                (ast.literal_eval(d) for d in node.args.defaults))) if node.args.defaults else {}
            signature = inspect.Signature(
                [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   default=defaults.get(n, inspect.Parameter.empty),
                                   annotation=t) for n, t in arguments.items()],
                return_annotation=result)
        self.__signature__ = signature
        code = compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]),
                       filename, 'exec')
        self.__code__ = next(c for c in code.co_consts
                             if isinstance(c, types.CodeType) and c.co_name == node.name)

    def __call__(self, *args, **kwargs):
        raise TypeError('SourceFormula is compilation metadata, not an evaluator')

    def with_annotations(self, annotations):
        """Change physical bindings without creating an executable Python body."""
        result = copy.copy(self)
        result.__annotations__ = dict(annotations)
        result.__signature__ = self.__signature__.replace(
            parameters=[p.replace(annotation=annotations.get(p.name, p.annotation))
                        for p in self.__signature__.parameters.values()],
            return_annotation=annotations.get('return', self.__signature__.return_annotation))
        return result


def is_formula(fn):
    return inspect.isfunction(fn) or isinstance(fn, SourceFormula)


def with_annotations(fn, annotations):
    """One annotation adapter for source descriptions and historical controls."""
    if isinstance(fn, SourceFormula):
        return fn.with_annotations(annotations)
    result = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__,
                                fn.__defaults__, fn.__closure__)
    result.__dict__.update(fn.__dict__)
    result.__annotations__ = dict(annotations)
    result.__module__ = fn.__module__
    return result


def formula_source(fn):
    return fn.source if isinstance(fn, SourceFormula) else inspect.getsource(fn)


def formula_location(fn):
    if isinstance(fn, SourceFormula):
        return fn.filename, fn.node.lineno
    return inspect.getsourcefile(fn), inspect.getsourcelines(fn)[1]
