"""IntEnum members in formulas (enums.plan): integers where the use is a number, member objects everywhere else.

A member becomes its integer as an operand of a comparison or of arithmetic, as an argument of a Cell whose
parameter is a number, as a subscript that is read, and in a display that a loop iterates when the loop variable
is used only in such ways. A parameter or Cell result traced as one IntEnum class is a C integer; where its
member is needed it is converted through its class. What reaches such a parameter or result from a value whose
class the text does not show goes through a check of its class.
"""
import enum
from types import SimpleNamespace

import pytest

from modelx_cython.enums import enum_safe, literals, plan
from modelx_cython.tracer import RuntimeRefInfo, RuntimeParamInfo


class Kind(enum.IntEnum):
    A = 1
    B = 2
    LOW = -3


class Leg(enum.IntEnum):
    CALL = 1
    PUT = 2


class Loose(enum.IntEnum):          # not enum-safe: its own == is kept as written
    X = 1

    def __eq__(self, other):
        return False

    __hash__ = enum.IntEnum.__hash__


class Ordered(enum.IntEnum):        # not enum-safe: its own ordering
    LOW = 2
    HIGH = 1

    def __lt__(self, other):
        return int(self) > int(other)


class Pretty(enum.IntEnum):         # not enum-safe: its own str and format
    M = 0

    def __str__(self):
        return self.name

    __format__ = lambda self, spec: self.name


class Texted(enum.IntEnum):         # not enum-safe: its value is not its integer
    A = 1

    def __new__(cls, number):
        member = int.__new__(cls, number)
        member._value_ = 'one'
        return member


def result(tp):
    return SimpleNamespace(value_type=tp, is_array=False)


def cell(result_type, **arg_types):
    return SimpleNamespace(arg_types=arg_types, ret_type=result(result_type))


REFS = {'m._c_P.Kind': RuntimeRefInfo(Kind), 'm._c_P.Leg': RuntimeRefInfo(Leg), 'm._c_P.Loose': RuntimeRefInfo(Loose),
        'm._c_Q.Other': RuntimeRefInfo(Kind), 'other._c_P.Kind': RuntimeRefInfo(Kind),
        'm._c_P.rate': RuntimeRefInfo(0.5), 'm._c_P.first': RuntimeRefInfo(Kind.A),
        'm._c_P.far': SimpleNamespace(mx_class='m._c_Q', type_=object)}
CELLS = {'m._c_P.f': cell(float, k=Kind),                       # the formula under test: k is a Kind
         'm._c_P._f_g': cell(float, t=int, k=Kind),             # a Cell with a Kind parameter, only compared
         'm._c_P._f_h': cell(float, t=int, x=float),            # numbers only
         'm._c_P._f_obj': cell(str, k=object),                  # an object parameter
         'm._c_P._f_kind': cell(Kind, t=int),                   # a Kind result
         'm._c_P._f_leg': cell(Leg),
         'm._c_Q._f_pick': cell(Kind),                          # a Cell of the Space that P.far holds
         '_m_P._mx_classes._c_O._f_pick': cell(Kind)}         # a Cell of P's child Space O


def formula(body, params='self, k', name='f'):
    return f'class _c_P:\n    def {name}({params}):\n        ' + body.replace('\n', '\n        ') + '\n'


def translate(body, cells=CELLS, **more):
    return literals('m', formula(body, **more), REFS, cells).split('\n', 2)[2][8:].replace('\n        ', '\n')


def test_enum_safe_needs_every_operation_of_int_and_the_integer_as_value():
    assert enum_safe(Kind) and enum_safe(Leg)
    assert not any(enum_safe(cls) for cls in (Loose, Ordered, Pretty, Texted, int, enum.Enum))


@pytest.mark.parametrize('body, expected', [
    ('return k == self.Kind.A', 'return k == 1'),                                   # a comparison
    ('return self.Kind.B != k or k < self.Kind.B', 'return 2 != k or k < 2'),
    ('return k in (self.Kind.A, self.Kind.B)', 'return k in (1, 2)'),               # right of `in`
    ('return k not in {self.Kind.A, self.Kind.B}', 'return k not in {1, 2}'),
    ('return self.g(t, self.Kind.B)', 'return self.g(t, 2)'),                       # an argument that is a number
    ('return self.g(t, k=self.Kind.B)', 'return self.g(t, k=2)'),
    ('return self.h(self.Kind.A, self.Kind.B)', 'return self.h(1, 2)'),
    ('return self.g(t, self.Kind.A if t else self.Kind.B)', 'return self.g(t, 1 if t else 2)'),
    ('return 10.0 * t + self.Kind.B', 'return 10.0 * t + 2'),                       # arithmetic
    ('return self.Kind.LOW ** 2', 'return (-3) ** 2'),
    ('return -(self.Kind.LOW)', 'return -((-3))'),
    ('return self.Kind.A if k else self.Kind.B', 'return 1 if k else 2'),           # the result is a number
    ('return T.loc[(self.Kind.A, k), "q"]', 'return T.loc[(1, k), "q"]'),           # a subscript that is read
    ('return rates[self.Kind.A] + rows[t:self.Kind.B]', 'return rates[1] + rows[t:2]'),
    ('return range(self.Kind.B), int(self.Kind.A)', 'return range(2), int(1)'),     # builtins that take a number
    ('return max(k, self.Kind.B) == self.Kind.B', 'return max(k, 2) == 2'),
    ('return sum(self.g(t, x) for x in (self.Kind.A, self.Kind.B))', 'for x in (1, 2))'),    # a loop of numbers
    ('for x in [self.Kind.A, self.Kind.B]:\n    y = self.g(t, x)', 'for x in [1, 2]:'),
    ('return [x == k for x in (self.Kind.A, self.Kind.B)]', 'for x in (1, 2)]'),
    ('x = self.Kind.A if t else self.Kind.B\nreturn self.g(t, x)', 'x = 1 if t else 2'),      # a local of numbers
    ('match k:\n    case self.Kind.A:\n        return 1.0', 'case 1:'),
])
def test_a_member_used_as_a_number_is_its_integer(body, expected):
    assert expected in translate(body)


@pytest.mark.parametrize('body', [
    'return (self.Kind.A, self.Kind.B)', 'return [self.Kind.A]', 'return {self.Kind.A: 1.0, "x": self.Kind.B}',
    'x = self.Kind.A\nreturn x.name', 'x = self.Kind.A if t else None\nreturn x',
    'return self.Kind.A.name, self.Kind.B.value, self.Kind.A.label',
    'return repr(self.Kind.A), str(self.Kind.A), format(self.Kind.A), ascii(self.Kind.A)',
    'return type(self.Kind.A), isinstance(self.Kind.A, self.Kind), getattr(self.Kind.B, "name"), id(self.Kind.A)',
    'return f"{self.Kind.A}|{self.Kind.A!r}|{self.Kind.B:>3}"', 'return "%s %r" % (self.Kind.A, self.Kind.B)',
    'return "%d" % self.Kind.A', 'return "{!r}".format(self.Kind.A)',
    'return self.label(self.Kind.A)', 'return self.obj(self.Kind.A)', 'return self.unknown(self.Kind.A)',
    'out = []\nout.append(self.Kind.A)\nreturn out', 'return d.get(self.Kind.A)',
    'd = {}\nd[self.Kind.A] = 1.0\nreturn d', 'return self.O[self.Kind.A].val()', 'return self.far[self.Kind.A]',
    'return x is self.Kind.A', 'return self.Kind.A is not x', 'return a is b',
    'return [x.name for x in (self.Kind.A, self.Kind.B)]',                  # a loop variable used as a member
    'return {x.name: self.g(t, x) for x in (self.Kind.A, self.Kind.B)}',
    'for x in [self.Kind.A, self.Kind.B]:\n    y = repr(x)',
    'return [isinstance(x, self.Kind) for x in (self.Kind.A, self.Kind.B)]',
    'return sorted((self.Kind.A, self.Kind.B)), max((self.Kind.A, self.Kind.B)).name',
    'y = max(x, self.Kind.B)', 'return fmt % self.Kind.A',
    'return self.Loose.X == k',                 # not enum-safe
    'return self.rate',                         # not an enum
    'return self.Other.A == k',                 # a Reference of another class
    'return self.Kind.MISSING == k',            # not a member
    'self = t\nreturn self.Kind.A == 1',        # `self` rebound
    'def inner(self):\n    return self.Kind.A == 1\nreturn inner',          # a function with its own `self`
])
def test_a_member_used_as_an_object_stays(body):
    assert translate(body, params='self, t') == body + '\n'


@pytest.mark.parametrize('body, expected', [
    ('return k.name', 'return _mx_sys._mx_enum_of(self.Kind, k).name'),                                  # a parameter
    ('return k.value + 1, k.bit_length()', 'return _mx_sys._mx_enum_of(self.Kind, k).value + 1, _mx_sys._mx_enum_of(self.Kind, k).bit_length()'),
    ('return repr(k), str(k), f"{k}"', 'return repr(_mx_sys._mx_enum_of(self.Kind, k)), str(_mx_sys._mx_enum_of(self.Kind, k)), f"{_mx_sys._mx_enum_of(self.Kind, k)}"'),
    ('return "%r" % k, "%s" % (k, 1)', 'return "%r" % _mx_sys._mx_enum_of(self.Kind, k), "%s" % (_mx_sys._mx_enum_of(self.Kind, k), 1)'),
    ('return type(k), isinstance(k, int)', 'return type(_mx_sys._mx_enum_of(self.Kind, k)), isinstance(_mx_sys._mx_enum_of(self.Kind, k), int)'),
    ('return (k, 1), [k], {k: 1.0}', 'return (_mx_sys._mx_enum_of(self.Kind, k), 1), [_mx_sys._mx_enum_of(self.Kind, k)], {_mx_sys._mx_enum_of(self.Kind, k): 1.0}'),
    ('return k is self.Kind.A', 'return _mx_sys._mx_enum_of(self.Kind, k) is self.Kind.A'),
    ('return self.label(k), self.obj(k)', 'return self.label(_mx_sys._mx_enum_of(self.Kind, k)), self.obj(_mx_sys._mx_enum_of(self.Kind, k))'),
    ('d = {}\nd[k] = 1.0', 'd = {}\nd[_mx_sys._mx_enum_of(self.Kind, k)] = 1.0'),
    ('return [lambda: k.name, lambda k: k.name]', 'return [lambda: _mx_sys._mx_enum_of(self.Kind, k).name, lambda k: k.name]'),
    ('return [k.name for k in names]', 'return [k.name for k in names]'),           # shadowed by a comprehension
    ('return self.kind(t).name', 'return _mx_sys._mx_enum_of(self.Kind, self.kind(t)).name'),            # a Cell result
    ('return self.O.pick().name', 'return _mx_sys._mx_enum_of(self.Kind, self.O.pick()).name'),          # of a child Space
    ('return self.far.pick().name', 'return _mx_sys._mx_enum_of(self.Kind, self.far.pick()).name'),      # of a Space in a Reference
    ('x = self.kind(t)\nreturn x.name', 'x = _mx_sys._mx_enum_of(self.Kind, self.kind(t))\nreturn x.name'),      # a local used as a member
    ('x = self.kind(t)\nreturn self.g(t, x) if x == self.Kind.A else 0.0',                  # a local of numbers
     'x = self.kind(t)\nreturn self.g(t, x) if x == 1 else 0.0'),
    ('k = k or self.Kind.A\nreturn k.name', 'k = k or 1\nreturn _mx_sys._mx_enum_of(self.Kind, k).name'),         # rebound to its class
    ('return self.first.name, self.first == k', 'return _mx_sys._mx_enum_of(self.Kind, self.first).name, self.first == k'),
    ('return k == self.Kind.A, self.g(t, k), k + 1, T[k], not k', 'return k == 1, self.g(t, k), k + 1, T[k], not k'),
    ('y = max(k, self.Kind.B)', 'y = max(_mx_sys._mx_enum_of(self.Kind, k), self.Kind.B)'),
])
def test_a_value_held_as_an_integer_is_converted_where_its_member_is_needed(body, expected):
    assert translate(body).startswith(expected)


def test_the_class_is_imported_where_no_reference_holds_it():
    source = 'class _c_R:\n    def f(self, k):\n        return k.name\n'
    assert (f"_mx_sys._mx_enum_of(__import__('importlib').import_module({Kind.__module__!r}).Kind, k).name"
            in literals('m', source, REFS, {'m._c_R.f': cell(str, k=Kind)}))


def test_the_parameter_of_an_item_space_is_converted():
    source = 'class _c_S:\n    def _f_val(self):\n        return self.leg.name, self.leg == 1\n'
    params = {'m._c_S': {'leg': RuntimeParamInfo(Leg.PUT)}}
    assert f'({Leg.__module__!r}).Leg, self.leg).name, self.leg == 1' in literals('m', source, REFS, CELLS, params)


def test_a_parameter_rebound_to_something_else_cannot_give_its_member():
    assert translate('k = t + 1\nreturn k == self.Kind.A') == 'k = t + 1\nreturn k == 1\n'
    with pytest.raises(TypeError, match=r'^P\.f: `k` needs the IntEnum member of a parameter that the formula rebinds'):
        translate('k = t + 1\nreturn k.name')


def test_a_module_without_enum_values_is_not_parsed():
    source = 'class _c_P:\n    def f(self):  return  self.Kind.A\n'
    assert literals('n', source, {'m._c_P.Kind': RuntimeRefInfo(Kind)}) is source


# ---------------------------------------------------------------- the class of what a converted value was given
CHECKED = '''\
class _c_P:
    def _f_nm(self, k):
        return k.name
    def _f_cmp(self, k):
        return k == self.Leg.PUT
    def _f_pick(self, t):
        if t > 5:
            return self.legs()[0]
        return self.Leg.PUT if t > 0 else self.Kind.A
    def _f_same(self, t):
        return self.Leg.PUT if t > 0 else self.pick(t)
    def _f_use(self, t):
        return self.pick(t).name
    def _f_calls(self, k, t):
        a = [self.nm(x) for x in (self.Kind.A, self.Kind.B)]
        b = [self.cmp(x) for x in (self.Kind.A, self.Kind.B)]
        c = [self.nm(x) for x in (self.Leg.CALL, self.Leg.PUT)]
        return self.nm(self.Leg.PUT), self.nm(self.Kind.A), self.nm(k), self.nm(self.legs()[0]), self.nm(t), self.cmp(t)
    def _f_through(self, j):
        return self.nm(j)
    def _f_start(self):
        return self.through(self.Kind.A), self.through(self.Leg.PUT)
    def nm(self, k):
        return self._f_nm(k)
'''
CHECKED_CELLS = {'m._c_P._f_nm': cell(str, k=Leg), 'm._c_P._f_cmp': cell(bool, k=Leg),
                 'm._c_P._f_pick': cell(Leg, t=int), 'm._c_P._f_same': cell(Leg, t=int),
                 'm._c_P._f_use': cell(str, t=int), 'm._c_P._f_calls': cell(object, k=Kind, t=int),
                 'm._c_P._f_through': cell(str, j=Leg), 'm._c_P._f_start': cell(object)}


def test_a_value_of_unknown_class_is_checked_where_its_receiver_converts():
    text = plan({'m': CHECKED}, REFS, CHECKED_CELLS)['m']
    check = '_mx_sys._mx_enum_member(self.Leg, '
    for expected in [
        'return _mx_sys._mx_enum_of(self.Leg, k).name',                                              # nm converts its parameter
        'return k == 2',                                                        # cmp only compares: any integer
        f'return {check}self.legs()[0])',                                       # pick's result is converted by use
        f'return {check}self.Leg.PUT if t > 0 else self.Kind.A)',               # a member of another class: raises
        'return 2 if t > 0 else self.pick(t)',                                  # of its own class: as integers
        f'a = [self.nm({check}x)) for x in (self.Kind.A, self.Kind.B)]',
        'b = [self.cmp(x) for x in (1, 2)]',
        'c = [self.nm(x) for x in (1, 2)]',
        f'return self.nm(2), self.nm({check}self.Kind.A)), self.nm({check}_mx_sys._mx_enum_of(self.Kind, k))), '
        f'self.nm({check}self.legs()[0])), self.nm({check}t)), self.cmp(t)',
        'return self.nm(j)',                                                    # through's parameter is of the class
        f'return self.through({check}self.Kind.A)), self.through(2)',           # so what reaches it is checked too
        'def nm(self, k):\n        return self._f_nm(k)',                       # the cache method is not a formula
    ]:
        assert expected in text, expected + '\n' + text
