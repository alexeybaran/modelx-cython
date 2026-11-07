"""Range proofs must cover both cache entries and every extra formula use."""
import ast
import pytest
from modelx_cython.cell_entries import DenseAxis,DenseEntry
from modelx_cython.dense_proofs import _attribute_counts,_rewrite


@pytest.mark.parametrize('public_guard,extra_use,expected',[
    ('0 <= t < 4',False,True),
    ('-1 <= t < 5',False,False),
    ('0 <= t < 4',True,False),
])
def test_public_cache_range_must_be_checked(public_guard,extra_use,expected):
    source=f'''class _c_Policy:
    @_mx_cy.cfunc
    def _f_value(self, t):
        return float(t)
    @_mx_cy.cfunc
    def _f_derived(self, t):
        if type(self) is _c_Policy:
            return _mx_cache_entry_0(self, t) + 1.0
        return self.value(t) + 1.0
    def value(self, t):
        if 0 <= t < 4:
            return self._f_value(t)
        else:
            raise IndexError()
    def derived(self, t):
        if {public_guard}:
            return self._f_derived(t)
        else:
            raise IndexError()
def _mx_cache_entry_0(self, t):
    if 0 <= t < 4:
        return self._f_value(t)
    else:
        raise IndexError()
def _mx_cache_entry_1(self, t):
    if 0 <= t < 4:
        return self._f_derived(t)
    else:
        raise IndexError()
'''
    if extra_use:source+='def unchecked(s, t):\n    return s._f_derived(t)\n'
    layouts={f'_mx_cache_entry_{i}':DenseEntry('_c_Policy',formula,
        (DenseAxis('t',0,4,'long long'),),False)
        for i,formula in enumerate(('_f_value','_f_derived'))}
    tree=ast.parse(source)
    result=_rewrite(source.encode(),tree,layouts,_attribute_counts(tree))
    assert (b'_mx_proven_axes' in result) is expected
