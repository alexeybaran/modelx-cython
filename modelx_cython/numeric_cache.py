"""Exact native keys for existing numeric dictionary Cells; no domain restriction."""
import ast
import re
import textwrap

TYPES = {'_mx_cy.longlong': 'long long', '_mx_cy.double': 'double', '_mx_cy.bint': 'bint'}


def support(tag, types, result, *, native=False):
    args = ', '.join(f'{tp} k{i}' for i, tp in enumerate(types))
    names = ', '.join(f'k{i}' for i in range(len(types)))
    fields = '\n'.join(f'    {tp} k{i}' for i, tp in enumerate(types))
    hashes = []
    for i, tp in enumerate(types):
        if tp == 'double':
            hashes.append(f'    bits.d = 0.0 if k{i} == 0.0 else k{i}\n    h = (h ^ bits.u) * <uint64_t>1099511628211')
        else:
            hashes.append(f'    h = (h ^ <uint64_t>k{i}) * <uint64_t>1099511628211')
    equality = ' and '.join(f'c.items[pos].k{i} == k{i}' for i in range(len(types)))
    copied = ', '.join(f'c.items[i].k{j}' for j in range(len(types)))
    setkeys = '\n'.join(f'    c.items[pos].k{i} = k{i}' for i in range(len(types)))
    key = 'entry.k0' if len(types) == 1 else '(' + ', '.join(f'entry.k{i}' for i in range(len(types))) + ')'
    source = f"""
cdef struct {tag}_entry:
    bint used
{fields}
    {result} value

cdef struct {tag}_cache:
    {tag}_entry* items
    size_t capacity
    size_t count

@cython.overflowcheck(False)
cdef inline uint64_t {tag}_hash({args}) noexcept:
    cdef uint64_t h = 1469598103934665603
    cdef _mx_bits bits
{chr(10).join(hashes)}
    h ^= h >> 32
    return h

cdef inline Py_ssize_t {tag}_find({tag}_cache* c, {args}) noexcept:
    cdef size_t pos
    if c.capacity == 0:
        return -1
    pos = {tag}_hash({names}) & (c.capacity - 1)
    while c.items[pos].used:
        if {equality}:
            return <Py_ssize_t>pos
        pos = (pos + 1) & (c.capacity - 1)
    return -1

cdef inline void {tag}_grow({tag}_cache* c) except *:
    cdef size_t capacity = 16 if c.capacity == 0 else c.capacity * 2
    cdef size_t i, pos
    cdef {tag}_entry* items
    if capacity <= c.capacity or capacity > <size_t>-1 // sizeof({tag}_entry):
        raise MemoryError()
    items = <{tag}_entry*>calloc(capacity, sizeof({tag}_entry))
    if items == NULL:
        raise MemoryError()
    for i in range(c.capacity):
        if c.items[i].used:
            pos = {tag}_hash({copied}) & (capacity - 1)
            while items[pos].used:
                pos = (pos + 1) & (capacity - 1)
            items[pos] = c.items[i]
    free(c.items)
    c.items = items
    c.capacity = capacity

cdef inline void {tag}_put({tag}_cache* c, {args}, {result} value) except *:
    cdef Py_ssize_t found = {tag}_find(c, {names})
    cdef size_t pos
    if found >= 0:
        c.items[found].value = value
        return
    if c.count >= c.capacity // 2:
        {tag}_grow(c)
    pos = {tag}_hash({names}) & (c.capacity - 1)
    while c.items[pos].used:
        pos = (pos + 1) & (c.capacity - 1)
{setkeys}
    c.items[pos].value = value
    c.items[pos].used = True
    c.count += 1

cdef inline dict {tag}_dictionary({tag}_cache* c):
    cdef dict result = {{}}
    cdef size_t i
    cdef {tag}_entry entry
    for i in range(c.capacity):
        if c.items[i].used:
            entry = c.items[i]
            result[{key}] = entry.value
    return result
"""
    if native:
        # Same exact-key/growth primitive; adapt only its numeric checked ABI.
        # Object conversion stays GIL-held. Successful numeric work owns no
        # Python temporaries and may serve the shared frame's nogil entries.
        source = source.replace(') noexcept:', ') noexcept nogil:')
        source = source.replace(f'cdef inline void {tag}_grow({tag}_cache* c) except *:',
                                f'cdef inline int {tag}_grow({tag}_cache* c) except -1 nogil:')
        source = source.replace(f'cdef inline void {tag}_put({tag}_cache* c, {args}, {result} value) except *:',
                                f'cdef inline int {tag}_put({tag}_cache* c, {args}, {result} value) except -1 nogil:')
        source = source.replace('        raise MemoryError()',
                                '        with gil:\n            raise MemoryError()')
        source = source.replace('    c.capacity = capacity\n',
                                '    c.capacity = capacity\n    return 0\n')
        source = source.replace('        return\n', '        return 0\n')
        source = source.replace('    c.count += 1\n', '    c.count += 1\n    return 0\n')
    return source


def transform(source, declarations, module, support_path):
    tree = ast.parse(source); lines = source.splitlines(keepends=True)
    edits, snippets, imports, owners, owner_code = [], [], [], [], []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in module.classes:
            continue
        info = module.classes[cls.name]; selected = []
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name not in info.cells:
                continue
            cell = info.cells[method.name]; a = method.args
            if (not cell.has_formula_def or cell.is_locked or not cell.has_args()
                    or cell.formula_is_generator or (cell.has_typeinfo() and cell.is_arrayable())
                    or a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.posonlyargs
                    or not any(ast.unparse(d) == '_mx_cy.ccall' for d in method.decorator_list)):
                continue
            args = a.args[1:]; types = [TYPES.get(ast.unparse(x.annotation)) if x.annotation else None for x in args]
            result = TYPES.get(ast.unparse(method.returns)) if method.returns else None
            if not args or not all(types) or not result:
                continue
            name = method.name; tag = '_mx_nc' + cls.name + '_' + name
            selected.append((name, tag));snippets.append(support(tag, types, result))
            owners.append(f'cdef class {tag}_owner:\n    cdef {tag}_cache data\n')
            restore_args = 'key' if len(types) == 1 else ', '.join(f'key[{i}]' for i in range(len(types)))
            owner_code.append(f"""@_mx_cy.cclass
class {tag}_owner:
    def __dealloc__(self):
        _mx_free(self.data.items)
    def __reduce__(self):
        return ({tag}_restore, ({tag}_dictionary(_mx_cy.address(self.data)),))

def {tag}_restore(state):
    owner: {tag}_owner = {tag}_owner()
    for key, value in state.items():
        {tag}_put(_mx_cy.address(owner.data), {restore_args}, value)
    return owner
""")
            imports += [tag + s for s in ('_owner', '_find', '_put', '_dictionary')]
            names = ', '.join(x.arg for x in args)
            key = args[0].arg if len(args) == 1 else '(' + names + ')'
            nonfinite = ' or '.join('not _mx_isfinite(' + x.arg + ')' for x, tp in zip(args, types) if tp == 'double') or 'False'
            owner = 'self._mx_nc_' + name
            c = owner + '.data'; d = 'self._v_' + name
            body = f"""_mx_pos: _mx_cy.Py_ssize_t
if {d} is None and ({nonfinite}):
    _mx_fallback = {{}} if {owner} is None else {tag}_dictionary(_mx_cy.address({c}))
    {d} = _mx_fallback
    {owner} = None
if {d} is not None:
    _mx_key = {key}
    _mx_found = {d}.get(_mx_key, _mx_cache_missing)
    if _mx_found is not _mx_cache_missing:
        return _mx_found
    val = self._f_{name}({names})
    {d}[_mx_key] = val
    return val
if {owner} is not None:
    _mx_pos = {tag}_find(_mx_cy.address({c}), {names})
    if _mx_pos >= 0:
        return {c}.items[_mx_pos].value
val = self._f_{name}({names})
# A recursive formula can grow the cache or switch it to dictionary storage.
if {d} is None:
    if {owner} is None:
        {owner} = {tag}_owner()
    {tag}_put(_mx_cy.address({c}), {names}, val)
else:
    {d}[{key}] = val
return val
"""
            edits.append((method.body[0].lineno - 1, method.end_lineno, textwrap.indent(body, ' ' * 8)))
        if selected:
            if any(isinstance(n, ast.FunctionDef) and n.name == '__dealloc__' for n in cls.body):
                raise ValueError('Existing destructor requires integration')
            head = re.search(r'^cdef class ' + re.escape(cls.name) + r'\(.*?\):$', declarations, re.M);assert head
            fields = '\n' + ''.join(f'    cdef {tag}_owner _mx_nc_{name}\n' for name, tag in selected)
            declarations = declarations[:head.end()] + fields + declarations[head.end():]

    if not snippets:
        return source, declarations
    for start, end, text in sorted(edits, reverse=True):lines[start:end] = [text]
    support_path.write_text('cimport cython\nfrom libc.stdint cimport uint64_t\nfrom libc.stdlib cimport calloc, free\ncdef union _mx_bits:\n    double d\n    uint64_t u\n' + '\n'.join(snippets + owners), encoding='utf-8')
    support_path.with_suffix('.py').write_text('import cython as _mx_cy\nfrom cython.cimports.libc.stdlib import free as _mx_free\n' + '\n'.join(owner_code), encoding='utf-8')
    package = module.fqname.rsplit('.', 1)[0]
    pre = f'from cython.cimports.{package}._mx_numeric_cache import ' + ', '.join(imports) + '\nfrom cython.cimports.libc.stdlib import free as _mx_free\nfrom cython.cimports.libc.math import isfinite as _mx_isfinite\n'
    declarations = f'from {package}._mx_numeric_cache cimport ' + ', '.join(x for x in imports if x.endswith('_owner')) + '\n' + declarations
    source = ''.join(lines);lines = source.splitlines(keepends=True)
    last_import = max((n.end_lineno for n in ast.parse(source).body if isinstance(n, (ast.Import, ast.ImportFrom))), default=0)
    lines.insert(last_import, pre)
    return ''.join(lines) + '\n_mx_cache_missing = object()\n', declarations
