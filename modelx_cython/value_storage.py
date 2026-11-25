"""S1: dense cache values that are not zero-filled when an item is allocated.

Every dense Cell cache of a compiled class is a value array (``_v_<cell>``) and a readiness flag array
(``_has_<cell>``) declared inline in the class. ``PyType_GenericAlloc`` zero-fills the whole object, and
the value arrays are most of its bytes. A value is read only after its flag is set, and a flag is set only
after its value is stored, so the values need no zeroing. The generator declares the value arrays as the
last members of the class, widest elements first, so that they form one adjacent region; at import a
probe instance gives their offsets to ``_mx_sys.pxd`` (``__mx_values_finish``), which checks that the
region holds those arrays and nothing else and then installs an allocator that zeroes every other byte.
The flags, every object field and the header are initialised exactly as before. Where the check fails or
the CPython C API is not available, nothing is installed and every byte is zeroed as before.
"""
from .consts import CY_MOD, HAS_PREF, MX_SYS_MOD, VAR_PREF

# Element widths that cannot need padding between adjacent arrays when widest come first.
_WIDE = ('double', 'long long')


def value_cells(cls_info):
    """The Cells of a class whose cache is a dense value array, in their declaration order."""
    cells = [cell for cell in cls_info.cells.values()
             if not cell.is_special() and cell.has_formula_def and cell.has_args()
             and cell.has_typeinfo() and cell.is_arrayable()]
    return sorted(cells, key=lambda cell: cell.get_rettype_expr(c_style=True) not in _WIDE)


def pxd_declarations(cls_info):
    """``cdef`` lines of the value arrays: the last attribute declarations of the class."""
    return [f'cdef {cell.get_array_decl_expr(c_style=True)} {VAR_PREF}{cell.name}\n' for cell in value_cells(cls_info)]


def py_declarations(cls_info):
    """Annotations of the value arrays, after every other annotation of the class body."""
    return [f'{VAR_PREF}{cell.name}: {cell.get_array_decl_expr()}\n' for cell in value_cells(cls_info)]


def check_last(class_block, cls_info):
    """Build-time check of the layout rule: the value arrays are the class's last attribute declarations."""
    attributes = [line.strip() for line in class_block.splitlines()
                  if line.strip().startswith('cdef ') and '(' not in line]
    expected = [line.strip() for line in pxd_declarations(cls_info)]
    if expected and attributes[len(attributes) - len(expected):] != expected:
        raise AssertionError(f'{cls_info.name}: dense value arrays must be the last members of the class')


def layout_function(cls_name):
    return f'_mx_values_layout_{cls_name}'


def _first(cell, prefix):
    return f'probe.{prefix}{cell.name}' + '[0]' * len(cell.dense_axes)


def functions(cls_info):
    """A module-level C function that registers the class's value region (``__mx_values_finish``)."""
    cells = value_cells(cls_info)
    if not cells:
        return ''
    name = cls_info.name
    lines = [f'@{CY_MOD}.cfunc',
             f'def {layout_function(name)}() -> {CY_MOD}.bint:',
             f'    probe: {name} = {name}.__new__({name})',
             f'    layout: {CY_MOD}.p_void = {MX_SYS_MOD}._mx_values_begin({name}, probe)']
    for cell in cells:
        value, flag = _first(cell, VAR_PREF), _first(cell, HAS_PREF)
        lines.append(f'    {MX_SYS_MOD}._mx_values_add(layout, {CY_MOD}.address({value}), '
                     f'{CY_MOD}.sizeof(probe.{VAR_PREF}{cell.name}), {CY_MOD}.sizeof({value}), '
                     f'{CY_MOD}.address({flag}), {CY_MOD}.sizeof(probe.{HAS_PREF}{cell.name}), {CY_MOD}.sizeof({flag}))')
    lines.append(f'    return {MX_SYS_MOD}._mx_values_finish(layout)\n')
    return '\n'.join(lines) + '\n'


def install(classes):
    """The module statement that installs the allocators: ``_mx_uninitialised_values`` names each class
    with dense values and whether its values are left unwritten at allocation."""
    names = [cls.name for cls in classes if value_cells(cls)]
    if not names:
        return ''
    entries = ', '.join(f"'{name}': {layout_function(name)}()" for name in names)
    return f'_mx_uninitialised_values = {{{entries}}}\n'
