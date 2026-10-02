"""The compiler shares dense cache storage only in explicit collect scopes."""

from modelx_cython.collect_reuse import transform


SOURCE = '''
import cython as _mx_cy
@_mx_cy.cclass
class _c_Projection:
    _mx_native: _mx_store_c_Projection
    def collect(self):
        return None
@_mx_cy.cclass
class _mx_store_c_Projection:
    pass
'''
DECLARATIONS = '''
cdef class _c_Projection:
    cdef _mx_store_c_Projection _mx_native
cdef class _mx_store_c_Projection:
    cdef unsigned char _has_fib[2001]
    cdef unsigned char _has_chain[2001]
'''


def test_collect_hook_reuses_only_declared_dense_cache(tmp_path):
    output, declarations = transform(
        SOURCE, DECLARATIONS, tmp_path / 'collect.json')
    assert declarations == DECLARATIONS
    assert 'context.get(template)' in output
    assert 'context[template] = self._mx_native' in output
    assert "owner._has_fib[0]" in output
    assert "owner._has_chain[0]" in output
    assert (tmp_path / 'collect.json').exists()


def test_descendant_without_collect_declaration_gets_dormant_lifecycle_hooks(tmp_path):
    plain = SOURCE.replace('def collect(self)', 'def other(self)')
    output, declarations = transform(
        plain, DECLARATIONS, tmp_path / 'collect.json')
    assert 'def _mx_collect_attach' in output
    assert 'def _mx_collect_release' in output
    assert 'context.get(template)' in output
    assert declarations == DECLARATIONS
    assert (tmp_path / 'collect.json').exists()
