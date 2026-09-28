"""Only explicit helper domains become semantic collect contracts."""

import ast

from modelx_cython.collect_context_semantics import derive


def formulas(source):
    cls = ast.parse(source).body[0]
    return {node.name[3:]: node for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith('_f_')}


def test_category_enum_member_table_key_and_argument_propagation():
    source = '''
class _c_Projection:
    def _f_sex(self):
        return self.lookup.category(self.data.points(), self.point_id, 'sex', self.Sex)
    def _f_schedule(self, t):
        if t == 0:
            return self.Schedule.INITIAL
        return self.Schedule.RENEWAL
    def _f_scenario(self):
        if self.scenario == -1:
            return self.lookup.key(self.data.rates(), self.point_id)
        return self.lookup.key(self.data.rates(), self.scenario)
    def _f_mortality(self, sex, age):
        return self.lookup.at(self.data.mortality(), (age, sex), 'rate')
    def _f_root(self, t):
        return self.mortality(self.sex(), t)
'''
    found = derive(formulas(source), {'sex', 'schedule', 'scenario', 'mortality'})
    assert found == {
        'sex': {'result': {'kind': 'enum_ref', 'name': 'Sex'}, 'arguments': {}},
        'schedule': {'result': {'kind': 'enum_ref', 'name': 'Schedule'},
                     'arguments': {}},
        'scenario': {'result': {'kind': 'table_key', 'table': 'rates'},
                     'arguments': {}},
        'mortality': {'result': None, 'arguments': {
            'sex': {'kind': 'enum_ref', 'name': 'Sex'}}},
    }


def test_mixed_returns_and_unrelated_callers_remain_untyped():
    source = '''
class _c_Projection:
    def _f_category(self):
        if self.flag:
            return self.lookup.category(self.data.points(), self.key, 'x', self.First)
        return self.lookup.category(self.data.points(), self.key, 'x', self.Second)
    def _f_first(self):
        return self.lookup.category(self.data.points(), self.key, 'x', self.First)
    def _f_consumer(self, value):
        return value
    def _f_root(self):
        self.consumer(self.first())
        return self.consumer(self.category())
'''
    assert derive(formulas(source), {'category', 'consumer'}) == {}


def test_semantic_contract_never_uses_sampled_value_or_name_guess():
    source = '''
class _c_Projection:
    def _f_sex(self):
        return self.data.category_value()
    def _f_schedule(self):
        return self.Schedule.INITIAL if self.flag else 'other'
'''
    assert derive(formulas(source), {'sex', 'schedule'}) == {}
