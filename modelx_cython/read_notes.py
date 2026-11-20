"""Notes of what arrives at each table read during the sample run (decisions from tracing, not from text form).

While the sample runs, the modules of the exported model are imported through a loader that wraps, in the formula
text only: every subscript load, every method call, every ``in`` and every loop source. The wrapper performs the
original operation unchanged and records, for that place in the text, what kind of value the receiver was, what kind
of key it took and what kind of value came back.  A pandas object is followed by identity: the frame a zero-argument
Cell returned, a row or a sub-table read from it, its index.

``NOTES``: (class, function, line, column, end line, end column, operation) -> set of observations.
The transformer reads them by position in the same text; a place with no note was never reached by the sample and a
place with more than one observation saw more than one kind of value: both stay as written.
"""
import ast
import enum
import importlib.abc
import importlib.machinery
import sys
from contextlib import contextmanager

NOTES = {}
COUNTS = {}        # the same key -> evaluations in the sample
SEGMENTS = {}       # the same key -> the source text of the node (checked against the transformer's node)
_TABLES, _SUBS, _ROWS, _INDEXES, _KEEP = {}, {}, {}, {}, []


def _pandas():
    return sys.modules.get('pandas')


def kind(value):
    pd = _pandas()
    if isinstance(value, bool) or type(value).__name__ == 'bool_':
        return 'bool'
    if isinstance(value, enum.Enum):
        return 'enum'
    if isinstance(value, int) or type(value).__module__ == 'numpy' and 'int' in type(value).__name__:
        return 'int'
    if isinstance(value, float) or type(value).__module__ == 'numpy' and 'float' in type(value).__name__:
        return 'float'
    if isinstance(value, str):
        return 'str'
    if pd is not None:
        if isinstance(value, pd.DataFrame):
            if id(value) in _TABLES:
                return 'table:' + _TABLES[id(value)]
            if id(value) in _SUBS:
                return 'sub:' + _SUBS[id(value)]
            return 'frame'
        if isinstance(value, pd.Series):
            return 'row:' + _ROWS[id(value)] if id(value) in _ROWS else 'series'
        if isinstance(value, (pd.core.indexing._LocIndexer, pd.core.indexing._AtIndexer)):
            inner = kind(value.obj)
            return 'loc:' + inner if inner not in ('frame', 'series') else 'loc'
        if isinstance(value, pd.Index):
            return 'index:' + _INDEXES[id(value)] if id(value) in _INDEXES else 'Index'
    return type(value).__name__


def _shape(key):
    if isinstance(key, tuple):
        return tuple(_shape(part) for part in key)
    found = kind(key)
    return 'num' if found in ('int', 'float', 'enum') else found


def _note(nid, *what):
    NOTES.setdefault(nid, set()).add(what)
    COUNTS[nid] = COUNTS.get(nid, 0) + 1


def _follow(value, origin):
    """Remember where a pandas object came from; its index belongs to it."""
    pd = _pandas()
    if pd is None:
        return
    if isinstance(value, pd.DataFrame):
        if id(value) not in _TABLES and id(value) not in _SUBS:
            (_SUBS if origin[0] == 'sub' else _TABLES)[id(value)] = origin[1]
            _KEEP.append(value)
        if id(value.index) not in _INDEXES:
            _INDEXES[id(value.index)] = ('sub:' if id(value) in _SUBS else 'table:') + origin[1]
            _KEEP.append(value.index)
    elif isinstance(value, pd.Series) and origin[0] == 'row' and id(value) not in _ROWS:
        _ROWS[id(value)] = origin[1]
        _KEEP.append(value)


def r_sub(nid, receiver, key, /):
    out = receiver[key]
    pd = _pandas()
    found = kind(receiver)
    if pd is not None and found.startswith('loc:') and not isinstance(key, (pd.Series, list, slice)) \
            and type(key).__name__ != 'ndarray':
        base = found.split(':')[-1]
        if isinstance(out, pd.Series) and (not isinstance(key, tuple) or all(not isinstance(p, str) for p in key)):
            _follow(out, ('row', base))
        elif isinstance(out, pd.DataFrame) and found.startswith('loc:table:'):
            _follow(out, ('sub', base))
    column = key[-1] if isinstance(key, tuple) and key and isinstance(key[-1], str) else key if isinstance(key, str) else None
    _note(nid, found, _shape(key), kind(out), column)
    return out


def r_call(nid, receiver, name, /, *args, **kwargs):
    out = getattr(receiver, name)(*args, **kwargs)
    pd = _pandas()
    if pd is not None and isinstance(out, pd.DataFrame) and not args and not kwargs \
            and id(out) not in _TABLES and id(out) not in _SUBS:
        owner = type(receiver).__name__
        _follow(out, ('table', (owner[3:] if owner.startswith('_c_') else owner) + '.' + name))     # a Cell's table
    _note(nid, kind(receiver), name, kind(out))
    return out


def r_in(nid, left, right):
    _note(nid, kind(right), _shape(left))
    return left in right


def r_it(nid, source):
    _note(nid, kind(source))
    return source


def r_v(nid, value):
    _note(nid, kind(value))
    return value


class _Wrap(ast.NodeTransformer):
    def __init__(self, source):
        self.source = source
        self.cls = self.fn = None

    def nid(self, node, op):
        key = (self.cls, self.fn, node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, op)
        SEGMENTS[key] = ast.get_source_segment(self.source, node)
        return ast.Constant(key)

    def call(self, name, node, *args, keywords=()):
        return ast.copy_location(ast.Call(ast.Name(name, ast.Load()), list(args), list(keywords)), node)

    def visit_ClassDef(self, node):
        outer, self.cls = self.cls, node.name
        self.generic_visit(node)
        self.cls = outer
        return node

    def visit_FunctionDef(self, node):
        if self.cls is None or self.fn is not None or not node.name.startswith('_f_'):
            return node                     # formulas only: methods ``_f_<cell>`` of a Space class
        self.fn = node.name
        self.generic_visit(node)
        self.fn = None
        return node

    def visit_Lambda(self, node):
        return node

    def visit_Subscript(self, node):
        nid = self.nid(node, 'sub')
        self.generic_visit(node)
        if not isinstance(node.ctx, ast.Load) or isinstance(node.slice, ast.Slice):
            return node
        return self.call('_mx_note_sub', node, nid, node.value, node.slice)

    def visit_Call(self, node):
        nid = self.nid(node, 'call')
        olds = list(node.args)
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in ('int', 'float', 'bool') and len(olds) == 1:
            node.args = [self.call('_mx_note_v', olds[0], self.nid(olds[0], 'v'), node.args[0])]
        if not isinstance(node.func, ast.Attribute) or any(isinstance(a, ast.Starred) for a in node.args) \
                or any(k.arg is None for k in node.keywords):
            return node
        return self.call('_mx_note_call', node, nid, node.func.value, ast.Constant(node.func.attr), *node.args,
                         keywords=node.keywords)

    def visit_Compare(self, node):
        nid = self.nid(node, 'in')
        self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            found = self.call('_mx_note_in', node, nid, node.left, node.comparators[0])
            return found if isinstance(node.ops[0], ast.In) else ast.copy_location(ast.UnaryOp(ast.Not(), found), node)
        return node

    def _source(self, holder):
        nid = self.nid(holder.iter, 'it')
        self.generic_visit(holder)
        holder.iter = self.call('_mx_note_it', holder.iter, nid, holder.iter)
        return holder

    visit_For = visit_comprehension = _source


def instrument(source, filename):
    tree = ast.fix_missing_locations(_Wrap(source).visit(ast.parse(source, filename)))
    return compile(tree, filename, 'exec', dont_inherit=True)


class _Loader(importlib.machinery.SourceFileLoader):
    def source_to_code(self, data, path, *, _optimize=-1):
        source = importlib.util.decode_source(data) if isinstance(data, bytes) else data
        return instrument(source, path)

    def get_code(self, fullname):       # never a cached byte-code file of the plain text
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, package):
        self.package = package

    def find_spec(self, fullname, path, target=None):
        if fullname != self.package and not fullname.startswith(self.package + '.'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader) \
                and not fullname.endswith('_mx_sys'):
            spec.loader = _Loader(spec.loader.name, spec.loader.path)
        return spec


@contextmanager
def recording(package):
    """Record the reads of the model package ``package`` while it is imported and run inside the block."""
    import builtins
    import importlib.util       # noqa: F401  (decode_source)
    for store in (NOTES, COUNTS, SEGMENTS, _TABLES, _SUBS, _ROWS, _INDEXES):
        store.clear()
    del _KEEP[:]
    hooks = {'_mx_note_sub': r_sub, '_mx_note_call': r_call, '_mx_note_in': r_in, '_mx_note_it': r_it, '_mx_note_v': r_v}
    finder = _Finder(package)
    sys.meta_path.insert(0, finder)
    for name, hook in hooks.items():
        setattr(builtins, name, hook)
    try:
        yield NOTES
    finally:
        sys.meta_path.remove(finder)
        for name in hooks:
            delattr(builtins, name)
        del _KEEP[:]


def as_json():
    """The notes as a JSON value (the build writes them next to the per-read report)."""
    out = []
    for key in sorted(NOTES, key=lambda k: tuple(str(x) for x in k)):
        out.append({'class': key[0], 'function': key[1], 'at': list(key[2:6]), 'op': key[6],
                    'text': SEGMENTS.get(key), 'seen': sorted(repr(n) for n in NOTES[key])})
    return out


def normal(text):
    """Source text of a node, as both parsers give it: without surrounding spaces and outer parentheses."""
    text = ' '.join((text or '').split())
    while text.startswith('(') and text.endswith(')') and _balanced(text[1:-1]):
        text = text[1:-1].strip()
    return text


def _balanced(text):
    depth = 0
    for ch in text:
        depth += ch in '([{'
        depth -= ch in ')]}'
        if depth < 0:
            return False
    return depth == 0


def by_text():
    """{(class, function): {text of a node: {'seen': observations, 'count': evaluations, 'ops': operations}}}.
    The transformer finds a note by the text of its node within the formula (the same text ran in the sample);
    places of one formula with the same text share their observations."""
    out = {}
    for key, seen in NOTES.items():
        entry = out.setdefault(key[:2], {}).setdefault(normal(SEGMENTS.get(key)), {'seen': set(), 'count': 0, 'ops': set()})
        entry['seen'] |= seen
        entry['count'] += COUNTS.get(key, 0)
        entry['ops'].add(key[6])
    return out


SELECTORS = ('series', 'list', 'ndarray', 'slice', 'frame', 'Index', 'set', 'dict', 'DataFrame', 'Series')


def selects_rows(shape):
    """A recorded key shape with a part that selects several rows (a mask, a list, a slice, a pandas object)."""
    if isinstance(shape, tuple):
        return any(selects_rows(part) for part in shape)
    return shape in SELECTORS or str(shape).startswith(('row:', 'sub:', 'table:', 'index:', 'loc'))


def table_read(op, seen):
    """Whether an observation is a read of a traced table (what the audit expects to be lowered or reported)."""
    receiver = seen[0]
    if op == 'sub':
        return receiver.startswith(('loc:', 'row:'))
    if op == 'call':
        return receiver.startswith('index:') and seen[1] in ('max', 'min', 'asof', 'searchsorted')
    return op in ('in', 'it') and receiver.startswith('index:')
