"""Hold a generated Collect's stable individual mapping within a private batch."""

import libcst as cst

from .scopes import CstFacts, Names, has_scope_effects
from .syntax import arguments, code, dotted_name, literal_value, local_assignment, single_statement


def _cached_collect(module, ref, sources, method, collect_contract, return_expr):
    target = module.classes.get(getattr(ref, 'decl_type_expr', None))
    if target is None or not ref.is_relative:
        return None
    source = sources.get(target.name)
    contract = collect_contract(source) if source is not None else None
    if contract is None or not contract[3]:
        return None
    domain = contract[2]
    cell = target.cells.get(domain)
    if cell is None or not cell.has_formula_def or cell.has_args():
        return None
    collector = method(source, 'collect')
    if collector is None or collector.decorators:
        return None
    params = collector.params
    if (params.posonly_params or params.kwonly_params or isinstance(params.star_arg, cst.Param)
            or params.star_kwarg or tuple(p.name.value for p in params.params) != ('self', *contract[1])):
        return None
    required = 0
    for param in params.params[1:]:
        if param.default is None:
            required += 1
            continue
        try:
            default = literal_value(param.default)
        except ValueError:
            return None
        if type(default) is not int or not -(1 << 63) <= default < 1 << 63:
            return None
    call = return_expr(collector)
    if not isinstance(call, cst.Call) or len(call.args) != 6:
        return None
    if (dotted_name(call.func) != '_mx_sys.collect_from_space'
            or any(arg.star or arg.keyword for arg in call.args[:-1])
            or dotted_name(call.args[0].value) != 'self'
            or arguments(call.args[1].value, f'self.{domain}', 0) is None
            or call.args[-1].keyword is None or call.args[-1].keyword.value != 'cache'
            or dotted_name(call.args[-1].value) != 'self._mx_collect_cache'):
        return None
    try:
        if literal_value(call.args[2].value) != contract[0] or literal_value(call.args[3].value) != contract[1]:
            return None
    except ValueError:
        return None
    coordinates = call.args[4].value
    if (not isinstance(coordinates, cst.Tuple) or any(not isinstance(x, cst.Element) for x in coordinates.elements)
            or tuple(dotted_name(x.value) for x in coordinates.elements) != contract[1]):
        return None
    return target.name, domain, required, len(contract[1])


class _HoldIndividuals(cst.CSTTransformer):
    def __init__(self, targets, shape, factory):
        self.targets, self.shape = targets, shape
        self.names = Names(factory)
        self.memo = self.names.reserve('_mx_factory_individuals')
        self.count = 0

    def leave_SimpleStatementLine(self, original, updated):
        stmt = single_statement(original)
        var = local_assignment(stmt)
        if var is None or not isinstance(stmt.value, cst.Subscript) or len(stmt.value.slice) != 1:
            return updated
        item, = stmt.value.slice
        if (not isinstance(item.slice, cst.Index) or item.slice.star
                or dotted_name(item.slice.value) != self.shape.parameter):
            return updated
        result = stmt.value.value
        if not isinstance(result, cst.Attribute) or result.attr.value != 'individual':
            return updated
        call = result.value
        args = arguments(call)
        if args is None:
            return updated
        parts = (dotted_name(call.func) or '').split('.')
        extra = {name for name, _ in self.shape.extra}
        if (len(parts) != 3 or parts[0] != 'self' or parts[2] != 'collect' or parts[1] not in self.targets
                or any(not isinstance(arg, cst.Name) or arg.value not in extra for arg in args)):
            return updated
        owner, domain, required, maximum = self.targets[parts[1]]
        if not required <= len(args) <= maximum:
            return updated
        receiver = f'self.{parts[1]}'
        mapping = self.names.reserve('_mx_individual_mapping')
        keys = self.names.reserve('_mx_individual_domain')
        entry = f'{self.memo}[{self.count}]'
        self.count += 1
        return cst.parse_statement(f'''\
if {self.memo} is None or type({receiver}) is not {owner}:
    {var.value} = {code(stmt.value)}
elif {entry} is not None:
    {var.value} = {entry}[{self.shape.parameter}]
else:
    {mapping} = {code(result)}
    {var.value} = {mapping}[{self.shape.parameter}]
    if {owner}.__dictoffset__ == 0 and {owner}.__flags__ & 256:
        {keys} = {receiver}.{domain}()
        if type({keys}) is type(()) and _mx_sys._mx_collect_native_coords({keys}):
            {entry} = {mapping}
''').with_changes(leading_lines=updated.leading_lines)


def lower(cls, node, shape, *, collect_contract, method, return_expr):
    spec = cls.module.spec.get_spec(cls.fqname)
    if not spec.get('factory_collect_hold_individuals', False):
        return node, '', 0
    factory = method(node, '_mx_space_formula')
    if (factory is None or factory.decorators or not isinstance(factory.body, cst.IndentedBlock)
            or not shape.extra or has_scope_effects(factory.body.body)):
        return node, '', 0
    facts = CstFacts(factory.body.body)
    names = Names(factory).names
    if {'type', 'tuple', '_mx_sys'} & names or any(facts.stores[name] for name in (
            'self', shape.parameter, *(name for name, _ in shape.extra))):
        return node, '', 0
    classes = {item.name.value: item for item in cls.module.visitor.wrapper.module.body
               if isinstance(item, cst.ClassDef)}
    targets = {name: match for name, ref in cls.refs.items() if (match := _cached_collect(
        cls.module, ref, classes, method, collect_contract, return_expr)) is not None and match[0] not in names}
    if not targets:
        return node, '', 0
    transform = _HoldIndividuals(targets, shape, factory)
    body = factory.body.visit(transform)
    if not transform.count:
        return node, '', 0
    kwonly = (*factory.params.kwonly_params, cst.Param(cst.Name(transform.memo), default=cst.Name('None')))
    star = factory.params.star_arg
    if star is cst.MaybeSentinel.DEFAULT:
        star = cst.ParamStar()
    replacement = factory.with_changes(body=body, params=factory.params.with_changes(kwonly_params=kwonly, star_arg=star))
    lines = tuple(replacement if line is factory else line for line in node.body.body)
    return node.with_changes(body=node.body.with_changes(body=lines)), transform.memo, transform.count
