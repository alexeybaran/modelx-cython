"""Native helper returns only at an already-double formula return boundary."""
import libcst as cst

from .helpers import module_name


class NumericReturns(cst.CSTTransformer):
    def __init__(self, cls, cell, providers):
        self.cls, self.cell, self.providers = cls, cell, providers
        self.contexts = {}
        self.operations = set()

    def native_key(self, key):
        if isinstance(key, cst.Integer):
            return -(1 << 63) <= key.evaluated_value < (1 << 63)
        if isinstance(key, cst.Name) and key.value in self.cell.params:
            return self.cell.get_argtype_expr(key.value, c_style=True) in ('long long', 'bint')
        if (isinstance(key, cst.Attribute) and isinstance(key.value, cst.Name)
                and key.value.value == 'self'):
            ref = self.cls.refs.get(key.attr.value)
            return ref is not None and ref.get_type_expr(c_style=True) in ('long long','bint')
        if (isinstance(key, cst.Call) and isinstance(key.func, cst.Name)
                and key.func.value in ('min','max') and len(key.args) >= 2
                and not any(a.keyword or a.star for a in key.args)):
            return all(self.native_key(a.value) for a in key.args)
        if (isinstance(key, cst.Call) and isinstance(key.func, cst.Attribute)
                and isinstance(key.func.value, cst.Name) and key.func.value.value == 'self'):
            cell = self.cls.cells.get(key.func.attr.value)
            return (cell is not None and not cell.called_with_kwargs and not cell.body_has_closure
                    and cell.get_rettype_expr(c_style=True) in ('long long', 'bint'))
        return False

    def leave_Return(self, before, after):
        if (self.cell.get_rettype_expr(c_style=True) != 'double'
                or (not self.cell.has_formula_def and self.cell.called_with_kwargs)):
            return after
        call = after.value
        if not (isinstance(call, cst.Call) and isinstance(call.func, cst.Attribute)
                and isinstance(call.func.value, cst.Name) and call.func.value.value == 'self'
                and 3 <= len(call.args) <= 4 and not any(a.keyword or a.star for a in call.args)):
            return after
        ref = self.cls.refs.get(call.func.attr.value)
        operation = getattr(ref, 'numeric_helper', None)
        if not operation or not ref.helper or not self.native_key(call.args[1].value):
            return after
        provider, _ = ref.helper
        self.operations.add(provider + '.' + ref.helper[1])
        alias = module_name(provider)
        context = f'_v_mx_context_{self.cell.name}'
        if context[len('_v_'):] in self.cls.cells:
            return after
        self.contexts[context] = provider
        self.providers.add(provider)
        return after.with_changes(value=call.with_changes(
            func=cst.Attribute(cst.Name(alias), cst.Name(operation)),
            args=(cst.Arg(call.func), cst.Arg(cst.parse_expression('self.'+context)), *call.args)))
