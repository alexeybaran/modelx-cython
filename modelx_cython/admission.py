"""Describe actual Cython representations without claiming nogil execution."""

import re

from .consts import FORMULA_PREF, MODULE_PREF, SPACE_PREF


def _lock_reason(source, kind, resource='table'):
    _, *parents, _, cls, cell = source.split('.')
    space = '.'.join([*(name[len(MODULE_PREF):] for name in parents),
                      cls[len(SPACE_PREF):]])
    name = cell[len(FORMULA_PREF):]
    return f'native {kind} {resource} read from {space}.{name} requires {space} in export locked_spaces'


def describe(module_infos):
    from modelx_cython.numeric_cache import primary_capacity
    result = {}
    for module in module_infos.values():
        for cls in module.classes.values():
            for cell in cls.cells.values():
                if cell.is_special():
                    continue
                formula_python = cell.formula_is_generator or cell.formula_has_captured_default or (
                    not cell.has_formula_def and cell.body_has_closure)
                public_python = cell.body_has_closure or cell.called_with_kwargs
                if not cell.has_formula_def:
                    cache = 'none'
                elif not cell.has_args():
                    cache = 'scalar'
                elif cell.has_typeinfo() and cell.is_arrayable():
                    cache = 'dense'
                elif cell.uses_native_numeric_cache:
                    cache = 'native-numeric-keys'
                else:
                    cache = 'dictionary'
                reasons = []
                if not cell.has_typeinfo():
                    reasons.append('formula not observed by the sample')
                if cell.formula_is_generator:
                    reasons.append('yield requires a Python method')
                if cell.formula_has_captured_default:
                    reasons.append('captured default uses the compiled Python signature')
                if cell.body_has_closure:
                    reasons.append('closure requires a Python public method')
                if cell.called_with_kwargs:
                    reasons.append('keyword call retains the Python public boundary')
                if cell.is_locked:
                    reasons.append('exported shared-Space lock retained')
                if cell.observed_native_binding is not None and cell.native_provider_binding is None:
                    reasons.append('native provider result requires a matching return annotation or direct forwarding from a typed Cell')
                reasons.extend(_lock_reason(source, 'cached', resource)
                               for source, resource in getattr(cell, 'unlocked_provider_sources', ()))
                reasons.extend(_lock_reason(source, kind)
                               for source, kind in getattr(cell, 'unlocked_table_reads', ()))
                result[cell.fqname] = {
                    'formula': 'python-method' if formula_python else 'cython-cfunction',
                    'public_entry': 'python-method' if public_python else 'cython-cpdef',
                    'inline_entry': getattr(cell, 'inline_entry', None),
                    'return_type': cell.get_rettype_expr(c_style=True),
                    'arguments': {name: cell.get_argtype_expr(name, c_style=True)
                                  for name in cell.params},
                    'cache': cache,
                    'automatic_policy': module.spec.automatic_optimizations,
                    'dense_window': [[int(axis.lower), int(axis.stop)] for axis in cell.dense_axes]
                    if cache == 'dense' else [],
                    'dense_logical_domain': [[int(axis.lower), int(axis.stop)] for axis in cell.dense_logical_axes]
                    if cache == 'dense' else [],
                    'associative_logical_domain': {
                        axis.key: [int(axis.lower), int(axis.stop)] for axis in cell.associative_logical_axes
                    } if cache in ('native-numeric-keys', 'dictionary') else {},
                    'numeric_primary_capacity': primary_capacity(cell)
                    if cache == 'native-numeric-keys' else 0,
                    'power_memo': bool(getattr(cell, 'power_memo', None)),
                    'dense_offsets': [int(offset) for offset in cell.cache_offsets]
                    if cache == 'dense' else [],
                    'exact_spill': cache == 'dense' and bool(cell.has_dense_spill),
                    'nogil_qualified': False,
                    'native_helpers': getattr(cell, 'native_helpers', []),
                    'native_numeric_helpers': getattr(cell, 'native_numeric_helpers', []),
                    'table_reads_typed': getattr(cell, 'table_reads_typed', 0),
                    'table_reads_untyped': getattr(cell, 'table_reads_untyped', []),
                    'table_reads_keys': getattr(cell, 'table_reads_keys', []),
                    'native_reductions': getattr(cell, 'native_reductions', 0),
                    'native_enum_comparisons': getattr(cell, 'native_enum_comparisons', 0),
                    'native_enum_constants': getattr(cell, 'native_enum_constants', 0),
                    'reasons': reasons or ['original Cython representation'],
                }
    return result


# ---------------------------------------------------------------- M1 table reads (W1.1)
TABLE_TYPE = 'modelx_helpers.immutable_inputs.Table'
TABLE_READ_ATTRS = ('loc', 'at', 'index', 'iterrows')
NOT_LOWERED = 'not lowered to a typed handle read; runs as a Python call'
ROUND2_FALLBACK = ('not lowered to a typed handle read; the formula also has round-2 native reads '
                   '({operations}), which fall back to this Python read where they do not apply')
TABLE_READ_WARNING = '{cell}: per-step table read is not typed: {read} (formula line {line}): {reason}'


def untyped_read_warning(cell, entry):
    """Build warning text for one per-step untyped read (an entry of table_reads_untyped)."""
    return TABLE_READ_WARNING.format(cell=cell, **entry)


class TableReadSite:
    """One pandas-spelled read on a traced helper Table in a formula (before lowering)."""
    __slots__ = ('node', 'kind', 'read', 'line')

    def __init__(self, node, kind, read, line):
        self.node, self.kind, self.read, self.line = node, kind, read, line

    def entry(self, per_step, reason=None):
        return {'read': self.read, 'kind': self.kind, 'line': self.line,
                'per_step': per_step, 'reason': reason or NOT_LOWERED}


def table_read_sites(node, resolver, cls_info):
    """The table reads of one formula FunctionDef, in source order. Read-only.

    A receiver is a traced helper Table when it is a self-rooted call to a Cell whose every
    sampled result is a helper Table (``resolver.table_results``), a ``self.<ref>`` whose value
    is a helper Table, or a local name assigned (anywhere in the formula) from such a
    receiver or from ``<receiver>.loc[...]`` / ``.at[...]`` (a sub-table or a row), or bound
    by ``for ... in <receiver>.iterrows()``. A site is ``<receiver>.loc|at|index|iterrows``
    (kind = the attribute) or ``<receiver>[...]`` (kind ``column`` on a table, ``row`` on a
    row local). Reads inside comprehensions and generators count. ``line`` is the line within
    the formula, 1 being the ``def`` line.
    """
    import libcst as cst
    from libcst.metadata import MetadataWrapper, PositionProvider
    from .usage import RESOLVED, _extract_self_chain

    tables = {name for name, kind in resolver.table_results.items() if kind == TABLE_TYPE}
    refs = {name for name, ref in getattr(cls_info, 'refs', {}).items()
            if getattr(ref, 'type_', None) is not None
            and f'{ref.type_.__module__}.{ref.type_.__qualname__}' == TABLE_TYPE}
    if not tables and not refs:
        return []

    def receiver(expr, local):
        if isinstance(expr, cst.Call) and not expr.args:
            chain = _extract_self_chain(expr.func)
            if isinstance(chain, list) and chain:
                resolved, name = resolver.resolve_chain(cls_info.fqname, chain)
                return resolved == RESOLVED and name in tables
            return False
        if isinstance(expr, cst.Attribute) and isinstance(expr.value, cst.Name):
            return expr.value.value == 'self' and expr.attr.value in refs
        return isinstance(expr, cst.Name) and expr.value in local

    def derived(expr, local):
        """A value read from a table receiver that is itself table-like (sub-table or row)."""
        if receiver(expr, local):
            return True
        return (isinstance(expr, cst.Subscript) and isinstance(expr.value, cst.Attribute)
                and expr.value.attr.value in ('loc', 'at') and receiver(expr.value.value, local))

    class Locals(cst.CSTVisitor):
        def __init__(self):
            self.assigned = []      # (name, value)
            self.rows = set()

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target.target, cst.Name):
                    self.assigned.append((target.target.value, node.value))

        def visit_AnnAssign(self, node):
            if isinstance(node.target, cst.Name) and node.value is not None:
                self.assigned.append((node.target.value, node.value))

        def _iterrows(self, target, iterable):
            if (isinstance(iterable, cst.Call) and isinstance(iterable.func, cst.Attribute)
                    and iterable.func.attr.value == 'iterrows' and isinstance(target, cst.Tuple)
                    and len(target.elements) == 2 and isinstance(target.elements[1].value, cst.Name)):
                self.assigned.append((target.elements[1].value.value, iterable.func.value))
                self.rows.add(target.elements[1].value.value)

        def visit_For(self, node):
            self._iterrows(node.target, node.iter)

        def visit_CompFor(self, node):
            self._iterrows(node.target, node.iter)

    found = Locals()
    node.visit(found)
    local, rows = set(), set()
    changed = True
    while changed:      # fixed point: locals assigned from other table locals
        changed = False
        for name, value in found.assigned:
            if name not in local and derived(value, local):
                local.add(name)
                changed = True
                if name in found.rows or not receiver(value, local - {name}):
                    rows.add(name)      # .loc/.at result or iterrows row: may be a row

    sites = []

    class Sites(cst.CSTVisitor):
        def visit_Attribute(self, expr):
            if expr.attr.value in TABLE_READ_ATTRS and receiver(expr.value, local):
                sites.append((expr, expr.attr.value))

        def visit_Subscript(self, expr):
            if receiver(expr.value, local):
                sites.append((expr, 'row' if isinstance(expr.value, cst.Name)
                              and expr.value.value in rows else 'column'))

    node.visit(Sites())
    if not sites:
        return []
    module = cst.Module(body=[node])
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)     # same node objects
    positions = wrapper.resolve(PositionProvider)
    first = positions[node.name].start.line
    parents = {}

    class Parents(cst.CSTVisitor):
        def on_visit(self, child):
            for grandchild in child.children:
                parents[id(grandchild)] = child
            return True

    node.visit(Parents())
    result = []
    for expr, kind in sites:
        shown = expr
        for _ in range(2):      # T.loc[...]; T.index.max(); k in T.index stays T.index
            parent = parents.get(id(shown))
            if isinstance(parent, (cst.Subscript, cst.Call)) and (
                    getattr(parent, 'value', None) is shown or getattr(parent, 'func', None) is shown):
                shown = parent
            elif isinstance(parent, cst.Attribute) and parent.value is shown:
                shown = parent
            else:
                break
        text = re.sub(r'\s*#[^\'"\n]*$', '', module.code_for_node(shown), flags=re.M)    # comments
        text = ' '.join(text.replace('( ', '(').replace('[ ', '[').split()).replace('( ', '(').replace('[ ', '[')
        result.append(TableReadSite(expr, kind, text, positions[expr].start.line - first + 1))
    result.sort(key=lambda site: site.line)
    return result
