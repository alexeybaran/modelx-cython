"""Reuse dense numeric cache owners only inside exported collect traversals.

Ordinary ItemSpaces keep their independent cache owners. The exported collect
helper calls these hooks after binding a transient policy Space and after its
outputs have been detached. A context belongs to one collect invocation, so
recursive or nested collect calls never share scratch storage accidentally.
"""

import ast
import json
import re


def transform(source, declarations, receipt_path):
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if not any(any(isinstance(item, ast.FunctionDef) and item.name == 'collect'
                   for item in cls.body) for cls in classes):
        return source, declarations

    selected = []
    for cls in classes:
        owners = [node.annotation.id for node in cls.body
                  if isinstance(node, ast.AnnAssign)
                  and isinstance(node.target, ast.Name)
                  and node.target.id == '_mx_native'
                  and isinstance(node.annotation, ast.Name)
                  and node.annotation.id.startswith('_mx_store_')]
        if len(owners) != 1:
            continue
        owner_type = owners[0]
        match = re.search(r'^cdef class ' + re.escape(owner_type)
                          + r':\s*\n(.*?)(?=^cdef class |\Z)',
                          declarations, re.M | re.S)
        if not match:
            raise ValueError('collect cache owner has no declaration')
        flags = re.findall(r'^    cdef unsigned char (_has_\w+)\[(\d+)\]$',
                           match.group(1), re.M)
        if not flags:
            continue
        reset = '\n'.join(
            '    _mx_collect_memset(_mx_cy.address(owner.%s[0]), 0, %s)'
            % (flag, length) for flag, length in flags)
        methods = '''
def _mx_collect_attach(self, context, template):
    owner: %s = context.get(template)
    if owner is None:
        return
%s
    self._mx_native = owner

def _mx_collect_release(self, context, template):
    if self._mx_native is not None:
        context[template] = self._mx_native
''' % (owner_type, reset)
        cls.body.extend(ast.parse(methods).body)
        selected.append({'class': cls.name, 'owner': owner_type,
                         'flags': [{'name': flag, 'length': int(length)}
                                   for flag, length in flags]})

    if not selected:
        return source, declarations
    tree.body.insert(0, ast.parse(
        'from cython.cimports.libc.string import memset as _mx_collect_memset').body[0])
    ast.fix_missing_locations(tree)
    receipt_path.write_text(json.dumps({'schema': 1, 'selected': selected}, indent=2),
                            encoding='utf-8')
    return ast.unparse(tree) + '\n', declarations
