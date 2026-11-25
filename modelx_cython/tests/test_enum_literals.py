"""Enum members in formulas (enums.plan): positions where that makes the speed, member objects everywhere else.

A parameter given members of one family is a C integer, the member's position. It is an integer in three uses:
compared by ``==``, ``!=``, ``in`` with members, values or a held name of its class; given to a Cell parameter
held for its class; a loop over a display of members. Every other use sees the member. What reaches a held
parameter from elsewhere goes through ``_mx_enum_code``.
"""
import enum
from types import SimpleNamespace

import pytest

from modelx_cython.enums import family, literals, plan, positions
from modelx_cython.tracer import RuntimeCellsInfo, RuntimeParamInfo, RuntimeRefInfo


class Kind(enum.IntEnum):
    A = 0
    B = 1
    C = 2


class Leg(enum.IntEnum):
    CALL = 0
    PUT = 1


class Gapped(enum.IntEnum):         # not a family: its values are not 0..n-1
    A = 1
    B = 2


class Loose(enum.IntEnum):          # not a family: its own ==
    X = 0

    def __eq__(self, other):
        return False

    __hash__ = enum.IntEnum.__hash__


class Ordered(enum.IntEnum):        # a family: its own ordering is seen where the member is used
    LOW = 0
    HIGH = 1

    def __lt__(self, other):
        return int(self) > int(other)


def result(tp):
    return SimpleNamespace(value_type=tp, is_array=False)


def held(result_type, classes, **arg_types):
    """A traced Cell whose parameters in ``classes`` are held as positions (the tracer types them as int)."""
    return SimpleNamespace(arg_types={name: int if name in classes else tp for name, tp in arg_types.items()},
                           ret_type=result(result_type), enum_codes=dict(classes))


def refs():
    return {'m._c_P.Kind': RuntimeRefInfo(Kind), 'm._c_P.Leg': RuntimeRefInfo(Leg), 'm._c_P.Loose': RuntimeRefInfo(Loose),
            'm._c_P.Ordered': RuntimeRefInfo(Ordered), 'm._c_Q.Kind': RuntimeRefInfo(Kind),
            'm._c_P.rate': RuntimeRefInfo(0.5), 'm._c_P.first': RuntimeRefInfo(Kind.A),
            'm._c_P.far': SimpleNamespace(mx_class='m._c_Q', type_=object),
            '_m_P._mx_classes._c_O.Kind': RuntimeRefInfo(Kind)}


def cells():
    return {'m._c_P._f_f': held(float, {'k': Kind}, t=int, k=Kind),       # the formula under test: k is a Kind
            'm._c_P._f_g': held(float, {'k': Kind}, t=int, k=Kind),       # a Cell with a Kind parameter
            'm._c_P._f_h': held(float, {}, t=int, x=float),               # numbers only
            'm._c_P._f_obj': held(str, {}, k=object),                     # an object parameter
            'm._c_P._f_o': held(float, {'w': Ordered}, w=Ordered),
            'm._c_Q._f_q': held(float, {'k': Kind}, k=Kind),              # a Cell of the Space that P.far holds
            '_m_P._mx_classes._c_O._f_q': held(float, {'k': Kind}, k=Kind)}     # a Cell of P's child Space O


STUBS = ('    def _f_g(self, t, k):\n        pass\n    def _f_h(self, t, x):\n        pass\n'
         '    def _f_obj(self, k):\n        pass\n    def _f_o(self, w):\n        pass\n'
         'class _c_Q:\n    def _f_q(self, k):\n        pass\n')       # the formulas of the other Cells


def formula(body, params='self, t, k', name='_f_f'):
    return f'class _c_P:\n    def {name}({params}):\n        ' + body.replace('\n', '\n        ') + '\n'


def translate(body, params='self, t, k', infos=None):
    infos = cells() if infos is None else infos
    sources = {'m': formula(body, params) + STUBS, '_m_P._mx_classes': 'class _c_O:\n    def _f_q(self, k):\n        pass\n'}
    text = plan(sources, refs(), infos)['m']
    return text[:text.index('    def _f_g(')].split('\n', 2)[2][8:].replace('\n        ', '\n')


def at(name, cls='Kind'):
    return f'_mx_sys._mx_enum_at(self.{cls}, {name})'


def test_a_family_compares_and_hashes_as_its_values():
    assert family(Kind) is int and family(Leg) is int and family(Ordered) is int
    assert not any(family(cls) for cls in (Gapped, Loose, int, enum.Enum, enum.IntEnum))
    assert positions(Kind) == ({'A': 0, 'B': 1, 'C': 2}, {0: 0, 1: 1, 2: 2})
    info = RuntimeRefInfo(Kind)
    assert info.enum_safe and info.enum_class is Kind and info.enum[2] == {'A': 0, 'B': 1, 'C': 2}
    assert not RuntimeRefInfo(Gapped).enum_safe


@pytest.mark.parametrize('body, expected', [
    ('return k == self.Kind.B', 'return k == 1'),                                   # compared with a member
    ('return self.Kind.B != k', 'return 1 != k'),
    ('return k == 2, k != 7', 'return k == 2, k != -1'),                           # with a value; of no member
    ('return k in (self.Kind.A, self.Kind.B)', 'return k in (0, 1)'),               # in a display of members
    ('return k not in {self.Kind.A, self.Kind.C}', 'return k not in {0, 2}'),
    ('return k == self.Kind.A or k == self.Kind.C', 'return k == 0 or k == 2'),
    ('return self.g(t, self.Kind.B), self.g(t, k=self.Kind.C)', 'return self.g(t, 1), self.g(t, k=2)'),
    ('return self.g(t, k) + self.g(t, 2)', 'return self.g(t, k) + self.g(t, 2)'),     # given to a held parameter
    ('return sum(self.g(t, x) for x in (self.Kind.A, self.Kind.B))', 'return sum(self.g(t, x) for x in (0, 1))'),
    ('for x in [self.Kind.A, self.Kind.B]:\n    y = self.g(t, x)', 'for x in [0, 1]:'),
    ('return [x == k for x in (self.Kind.A, self.Kind.B)]', 'return [x == k for x in (0, 1)]'),
    ('return self.far.q(self.Kind.B), self.O.q(self.Kind.C)', 'return self.far.q(1), self.O.q(2)'),
])
def test_a_held_value_is_an_integer_where_members_are_told_apart(body, expected):
    assert translate(body).startswith(expected)


@pytest.mark.parametrize('body, expected', [
    ('return k.name, k.value + 1', f'return {at("k")}.name, {at("k")}.value + 1'),
    ('return k + 1, T[k], not k, -k', f'return {at("k")} + 1, T[{at("k")}], not {at("k")}, -{at("k")}'),
    ('return k < self.Kind.B, k == -1', f'return {at("k")} < self.Kind.B, {at("k")} == -1'),
    ('return repr(k), str(k), f"{k}"', f'return repr({at("k")}), str({at("k")}), f"{{{at("k")}}}"'),
    ('return type(k), isinstance(k, int), k is self.Kind.A', f'return type({at("k")}), isinstance({at("k")}, int), {at("k")} is self.Kind.A'),
    ('return (k, 1), [k], {k: 1.0}', f'return ({at("k")}, 1), [{at("k")}], {{{at("k")}: 1.0}}'),
    ('return self.label(k), self.obj(k), self.h(t, k)', f'return self.label({at("k")}), self.obj({at("k")}), self.h(t, {at("k")})'),
    ('d = {}\nd[k] = 1.0', f'd = {{}}\nd[{at("k")}] = 1.0'),
    ('return [lambda: k.name, lambda k: k.name]', f'return [lambda: {at("k")}.name, lambda k: k.name]'),
    ('return [k.name for k in names]', 'return [k.name for k in names]'),           # shadowed by a comprehension
    ('return k', f'return {at("k")}'),
    ('return max(k, self.Kind.B), k == other', f'return max({at("k")}, self.Kind.B), {at("k")} == other'),
    ('return self.Kind.A <= k <= self.Kind.B', f'return self.Kind.A <= {at("k")} <= self.Kind.B'),
    ('match k:\n    case self.Kind.A:\n        return 1.0', f'match {at("k")}:\n    case self.Kind.A:'),
])
def test_every_other_use_of_a_held_value_sees_its_member(body, expected):
    assert translate(body).startswith(expected)


@pytest.mark.parametrize('body', [
    'return (self.Kind.A, self.Kind.B)', 'return [self.Kind.A]', 'return {self.Kind.A: 1.0, "x": self.Kind.B}',
    'return 10.0 * t + self.Kind.B', 'return rates[self.Kind.A], range(self.Kind.B)',       # numbers: members
    'return self.Kind.A == self.Kind.B, t == self.Kind.A',          # no held value: members compare as in modelx
    'x = self.Kind.A\nreturn x.name', 'return self.Kind.A.name, self.Kind.B.value',
    'return repr(self.Kind.A), f"{self.Kind.A!r}", "%s" % self.Kind.A',
    'return self.label(self.Kind.A), self.obj(self.Kind.A), self.h(t, self.Kind.A)',
    'out = []\nout.append(self.Kind.A)\nreturn out', 'return self.O[self.Kind.A].val()',
    'return x is self.Kind.A, a is b',
    'return [x.name for x in (self.Kind.A, self.Kind.B)]',                  # a loop variable used as a member
    'for x in [self.Kind.A, self.Kind.B]:\n    y = repr(x)',
    'return [isinstance(x, self.Kind) for x in (self.Kind.A, self.Kind.B)]',
    'return self.Loose.X == self.Kind.A, self.rate', 'return self.Kind.MISSING == t',
    'def inner(self):\n    return self.Kind.A == 1\nreturn inner',
])
def test_a_member_used_as_an_object_stays(body):
    infos = cells()
    infos['m._c_P._f_f'] = held(float, {}, t=int)
    assert translate(body, params='self, t', infos=infos) == body + '\n'


def test_what_reaches_a_held_parameter_from_elsewhere_is_converted_at_the_cell():
    code = "_mx_sys._mx_enum_code(self.Kind, {}, False, 'P.g')"
    assert translate('return self.g(t, self.first), self.g(t, x)') == \
        f'return self.g(t, {code.format("self.first")}), self.g(t, {code.format("x")})\n'
    assert translate('x = self.Kind.A\nreturn self.g(t, x)') == f'x = self.Kind.A\nreturn self.g(t, {code.format("x")})\n'
    assert translate('return {x.name: self.g(t, x) for x in (self.Kind.A, self.Kind.B)}') == \
        f'return {{x.name: self.g(t, {code.format("x")}) for x in (self.Kind.A, self.Kind.B)}}\n'
    assert translate('return self.o(k)') == f"return self.o(_mx_sys._mx_enum_code(self.Ordered, {at('k')}, False, 'P.o'))\n"
    assert translate('return self.far.q(x)') == "return self.far.q(_mx_sys._mx_enum_code(self.far.Kind, x, False, 'Q.q'))\n"


CHAIN = '''\
class _c_P:
    def _f_nm(self, k):
        return k.name
    def _f_cmp(self, k):
        return k == self.Kind.B
    def _f_through(self, j):
        return self.nm(j), self.cmp(j)
    def _f_start(self, t):
        return self.through(t), self.cmp(t), self.nm(self.Kind.A)
    def nm(self, k):
        return self._f_nm(k)
'''


def test_a_parameter_whose_member_is_used_or_given_on_to_one_is_strict():
    infos = {'m._c_P._f_nm': held(str, {'k': Kind}, k=Kind), 'm._c_P._f_cmp': held(bool, {'k': Kind}, k=Kind),
             'm._c_P._f_through': held(str, {'j': Kind}, j=Kind), 'm._c_P._f_start': held(object, {}, t=object)}
    text = plan({'m': CHAIN}, refs(), infos)['m']
    for expected in [f'return {at("k")}.name', 'return k == 1', 'return self.nm(j), self.cmp(j)',
                     "return self.through(_mx_sys._mx_enum_code(self.Kind, t, True, 'P.through')), "
                     "self.cmp(_mx_sys._mx_enum_code(self.Kind, t, False, 'P.cmp')), self.nm(0)",
                     'def nm(self, k):\n        return self._f_nm(k)']:     # the cache method is not a formula
        assert expected in text, expected + '\n' + text
    assert {key.rpartition('.')[2]: info.enum_gate for key, info in infos.items() if hasattr(info, 'enum_gate')} == {
        '_f_nm': {'k': ('Kind', True)}, '_f_cmp': {'k': ('Kind', False)}, '_f_through': {'j': ('Kind', True)}}


@pytest.mark.parametrize('body, strict', [
    ('return 10.0 * t + k, -k, k < 2, T[k], f"{k}", 1.0 if k else 0.0', False),   # the member acts as its value
    ('return [x for x in T if k]', False),
    ('return k.name', True), ('return repr(k)', True), ('return k is None', True), ('return k', True),
    ('return self.h(t, k)', True), ('d = {}\nd[k] = 1.0', True), ('return k or 1', True),
])
def test_a_parameter_is_strict_where_its_member_shows_its_class(body, strict):
    infos = cells()
    translate(body, infos=infos)
    assert infos['m._c_P._f_f'].enum_gate == {'k': ('Kind', strict)}


def test_a_strict_parameter_that_the_sample_gave_a_plain_value_is_an_object():
    infos = {'m._c_P._f_nm': held(str, {'k': Kind}, k=Kind), 'm._c_P._f_cmp': held(bool, {'k': Kind}, k=Kind),
             'm._c_P._f_through': held(str, {'j': Kind}, j=Kind), 'm._c_P._f_start': held(object, {}, t=object)}
    infos['m._c_P._f_nm'].enum_plain = infos['m._c_P._f_cmp'].enum_plain = {'k'}     # given 1 as well as Kind.B
    text = plan({'m': CHAIN}, refs(), infos)['m']
    assert 'return k.name' in text and 'return k == 1' in text and f"return self.nm({at('j')}), self.cmp(j)" in text
    assert infos['m._c_P._f_nm'].arg_types['k'] is object and infos['m._c_P._f_cmp'].enum_gate == {'k': ('Kind', False)}
    assert infos['m._c_P._f_through'].enum_gate == {'j': ('Kind', True)}      # it gives nm the member


@pytest.mark.parametrize('params, body', [
    ('self, t, k', 'k = t + 1\nreturn k == self.Kind.A'),                 # bound again
    ('self, t, k=None', 'return k is None or k == self.Kind.A'),          # a default that is no member
    ('self, t, k=7', 'return k == self.Kind.A'),
    ('self, t, k', 'return [self for self in t if k == self.Kind.A]'),     # seen where `self` is another
])
def test_a_parameter_that_cannot_be_held_is_an_object(params, body):
    infos = cells()
    assert translate(body, params, infos) == body + '\n'
    assert infos['m._c_P._f_f'].arg_types['k'] is object and not infos['m._c_P._f_f'].enum_codes


def test_a_parameter_of_an_uncached_formula_with_a_closure_is_an_object():
    infos = cells()          # its public method, the formula itself, stays a Python method with untyped parameters
    infos['m._c_P.u'] = held(float, {'k': Kind}, t=int, k=Kind)
    body = 'return sum(t for _ in (1, 2)) if k == self.Kind.A else 0.0'
    assert translate(body, infos=infos) == f"return sum(t for _ in (1, 2)) if k == 0 else 0.0\n"
    source = formula(body, name='u') + STUBS
    assert literals('m', source, refs(), infos) == source and infos['m._c_P.u'].arg_types['k'] is object


def test_a_parameter_whose_space_has_no_reference_to_its_class_or_given_by_star_is_an_object():
    infos = cells()
    infos['m._c_R.f'] = held(float, {'k': Kind}, k=Kind)
    source = 'class _c_R:\n    def f(self, k):\n        return k.name\n' + STUBS
    assert literals('m', source, refs(), infos) == source and infos['m._c_R.f'].arg_types['k'] is object
    assert translate('return self.g(*args)', infos=infos) == 'return self.g(*args)\n'
    assert infos['m._c_P._f_g'].arg_types['k'] is object


def test_a_default_that_is_a_member_value_is_its_position():
    assert translate('return k == self.Kind.C', 'self, t, k=2').startswith('return k == 2')
    assert literals('m', formula('return k', 'self, t, k=1') + STUBS, refs(), cells()).startswith(
        'class _c_P:\n    def _f_f(self, t, k=1):')


def test_a_module_without_held_parameters_is_not_parsed():
    source = 'class _c_P:\n    def f(self):  return  self.Kind.A == 1\n'
    assert literals('m', source, refs(), {'m._c_P.f': held(bool, {})}) is source


def test_the_tracer_holds_members_of_one_family_and_their_values_as_positions():
    def f(self, t, kind, leg, mixed, gapped, label):
        return Kind.A

    def trace(*values):
        names = ('self', 't', 'kind', 'leg', 'mixed', 'gapped', 'label')
        return SimpleNamespace(func=f, funcname='m._c_P._f_f', arg_vals=dict(zip(names, (None, *values))),
                               ret_val=Kind.A)
    info = RuntimeCellsInfo([trace(0, Kind.A, Leg.PUT, Kind.A, Gapped.A, 3), trace(1, 2, Leg.CALL, Leg.PUT, Gapped.B, 4)])
    assert info.enum_codes == {'kind': Kind, 'leg': Leg}       # mixed: two classes; label: plain integers only
    assert info.arg_types['kind'] is int and (info.min_args['kind'], info.max_args['kind']) == (0, 2)
    assert info.arg_types['mixed'] is object and info.arg_types['gapped'] is object and info.arg_types['label'] is int
    assert info.ret_type.value_type is object                   # a member is returned as itself
    assert RuntimeRefInfo(Kind.B).type_ is object and RuntimeParamInfo(Leg.PUT).type_ is object


# ---------------------------------------------------------------- text-valued families
needs_strenum = pytest.mark.skipif(not hasattr(enum, 'StrEnum'), reason='enum.StrEnum is of Python 3.11')
if hasattr(enum, 'StrEnum'):
    class Claim(enum.StrEnum):
        DEATH = 'DEATH'
        LAPSE = 'LAPSE'
        MAT = 'MATURITY'

    class Sex(enum.StrEnum):
        M = 'M'
        F = 'F'


def text(body, params='self, t, kind'):
    infos = {'m._c_P.f': held(float, {'kind': Claim}, t=int, kind=Claim),
             'm._c_P._f_claims': held(float, {'kind': Claim}, t=int, kind=Claim),
             'm._c_P._f_rate': held(float, {'s': Sex}, t=int, s=Sex), 'm._c_P._f_sex': held(str, {})}
    text_refs = {'m._c_P.Claim': RuntimeRefInfo(Claim), 'm._c_P.Sex': RuntimeRefInfo(Sex),
                 'm._c_P.default_kind': RuntimeRefInfo('LAPSE')}
    stubs = ('    def _f_claims(self, t, kind):\n        pass\n    def _f_rate(self, t, s):\n        pass\n'
             '    def _f_sex(self):\n        pass\n')
    out = literals('m', formula(body, params=params, name='f') + stubs, text_refs, infos)
    return out[:out.index('    def _f_claims(')].split('\n', 1)[1][4:].replace('\n        ', '\n')


def text_body(body):
    return text(body).split('\n', 1)[1]


@needs_strenum
def test_a_text_valued_family_holds_positions():
    assert family(Claim) is str and family(Sex) is str
    assert positions(Claim) == ({'DEATH': 0, 'LAPSE': 1, 'MAT': 2}, {'DEATH': 0, 'LAPSE': 1, 'MATURITY': 2})

    def f(self, t, kind, sex, label):
        return 0.0

    def trace(*values):
        names = ('self', 't', 'kind', 'sex', 'label')
        return SimpleNamespace(func=f, funcname='m._c_P._f_f', arg_vals=dict(zip(names, (None, *values))),
                               ret_val=0.0)
    info = RuntimeCellsInfo([trace(0, Claim.DEATH, 'M', 'a'), trace(1, 'MATURITY', 'F', Claim.LAPSE)])
    assert info.enum_codes == {'kind': Claim}           # sex was given texts only; label a text that is no member
    assert info.arg_types['kind'] is int and (info.min_args['kind'], info.max_args['kind']) == (0, 2)
    assert info.arg_types['sex'] is str and info.arg_types['label'] is object


@needs_strenum
@pytest.mark.parametrize('body, expected', [
    ('return kind == self.Claim.DEATH', 'return kind == 0'),                        # compared with a member
    ('return kind != "LAPSE"', 'return kind != 1'),                                  # or with its text
    ('return kind in (self.Claim.MAT, "LAPSE")', 'return kind in (2, 1)'),
    ('return kind == "NOPE"', 'return kind == -1'),                                  # a text that no member has
    ('return self.claims(t, self.Claim.MAT) + self.claims(t, "DEATH")', 'return self.claims(t, 2) + self.claims(t, 0)'),
    ('return self.claims(t, kind) + self.claims(t, kind=self.Claim.LAPSE)', 'return self.claims(t, kind) + self.claims(t, kind=1)'),
    ('return sum(self.claims(t, k) for k in (self.Claim.DEATH, self.Claim.LAPSE))', 'return sum(self.claims(t, k) for k in (0, 1))'),
    ('return self.claims(t, self.default_kind)',                                    # a text that is known when it runs
     "return self.claims(t, _mx_sys._mx_enum_code(self.Claim, self.default_kind, False, 'P.claims'))"),
    ('return self.rate(t, self.sex())', "return self.rate(t, _mx_sys._mx_enum_code(self.Sex, self.sex(), False, 'P.rate'))"),
    ('return kind.name + kind', f'return {at("kind", "Claim")}.name + {at("kind", "Claim")}'),
    ('return kind < self.Claim.MAT, kind == 0', f'return {at("kind", "Claim")} < self.Claim.MAT, {at("kind", "Claim")} == 0'),
    ('return f"{kind}", self.Claim.MAT == "MATURITY"', f'return f"{{{at("kind", "Claim")}}}", self.Claim.MAT == "MATURITY"'),
])
def test_a_text_valued_member_is_its_position_where_members_are_told_apart(body, expected):
    assert text_body(body).startswith(expected)


@needs_strenum
def test_a_text_default_of_a_held_parameter_is_its_position():
    assert text('return kind == self.Claim.MAT', params='self, t, kind="MATURITY"').startswith(
        'def f(self, t, kind=2):\nreturn kind == 2')
