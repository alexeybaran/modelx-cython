import libcst as cst

from modelx_cython.cell_entries import EntryModule, FormulaEntry
from modelx_cython.entry_pipeline import _sibling_accessor_uses


def test_only_direct_calls_on_unchanged_extension_receivers_are_discounted():
    source = '''
@_mx_cy.cclass
class A(_mx_sys.BaseSpace):
    @_mx_cy.ccall
    def value(self, t):
        return self._f_value(t) + self.other._f_value(t)
@_mx_cy.cclass
class B(_mx_sys.BaseSpace):
    @_mx_cy.ccall
    def value(self, t):
        return self._f_value(t)
@_mx_cy.cclass
class C(_mx_sys.BaseSpace):
    @_mx_cy.ccall
    def value(self, t):
        self = self.other
        return self._f_value(t)
@_mx_cy.cclass
class D(A):
    @_mx_cy.ccall
    def value(self, t):
        return self._f_value(t)
def _mx_cache_entry_0(self, t):
    return self._f_value(t)
'''
    entry = FormulaEntry('B', '_f_value', ('t',), True, None)
    owners, totals, own = _sibling_accessor_uses(EntryModule(cst.parse_module(source)), {'_mx_cache_entry_0': entry})
    assert owners == {'A', 'B', 'C'}
    assert totals['_f_value'] == 3
    assert own['A', '_f_value'] == 1
    assert own['B', '_f_value'] == 2
    assert own['C', '_f_value'] == 0
