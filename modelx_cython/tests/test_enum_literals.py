"""IntEnum members held in References become integer literals before any other pass (enums.literals)."""
import enum
from types import SimpleNamespace

import pytest

from modelx_cython.enums import literals


class Kind(enum.IntEnum):
    A = 1
    B = 2
    LOW = -3


class Loose(enum.IntEnum):          # not enum-safe: its own == is kept as written
    X = 1

    def __eq__(self, other):
        return False

    __hash__ = enum.IntEnum.__hash__


def info(cls):
    from modelx_cython.tracer import RuntimeRefInfo
    return RuntimeRefInfo(cls)


REFS = {'m._c_P.Kind': info(Kind), 'm._c_P.Loose': info(Loose), 'm._c_Q.Other': info(Kind),
        'other._c_P.Kind': info(Kind), 'm._c_P.rate': info(0.5)}


def translate(body, cls='_c_P'):
    source = f'class {cls}:\n    def f(self, k):\n        ' + body.replace('\n', '\n        ') + '\n'
    return literals('m', source, REFS)


@pytest.mark.parametrize('body, expected', [
    ('return k == self.Kind.A', 'return k == 1'),
    ('return self.g(t, self.Kind.B)', 'return self.g(t, 2)'),
    ('return k in (self.Kind.A, self.Kind.B)', 'return k in (1, 2)'),
    ('for x in (self.Kind.A, self.Kind.B):\n    pass', 'for x in (1, 2):'),
    ('return sum(self.g(t, x) for x in (self.Kind.A, self.Kind.B))', 'for x in (1, 2))'),
    ('return self.Kind.A if k else self.Kind.B', 'return 1 if k else 2'),
    ('return self.Kind.LOW ** 2', 'return (-3) ** 2'),
    ('return (self.Kind.LOW)', 'return ((-3))'),
    ('return [lambda: self.Kind.A]', 'lambda: 1'),
])
def test_members_become_literals(body, expected):
    assert expected in translate(body)


@pytest.mark.parametrize('body, expected', [
    ('return (self.Kind.A, self.Kind.B)', 'return (self.Kind.A, self.Kind.B)'),    # a container keeps members
    ('return [self.Kind.A, k == self.Kind.B]', 'return [self.Kind.A, k == 2]'),
    ('return {self.Kind.A: 1.0, "x": self.Kind.B}', 'return {self.Kind.A: 1.0, "x": self.Kind.B}'),
    ('return (self.Kind.A, self.Kind.B) if k else (self.Kind.A,)',
     'return (self.Kind.A, self.Kind.B) if k else (self.Kind.A,)'),
    ('legs = (self.Kind.A,)\nfor x in legs:\n    pass', 'legs = (self.Kind.A,)'),
    ('return {x for x in (self.Kind.A, self.Kind.B)}', 'for x in (1, 2)}'),    # iterated in place
    ('return k in {self.Kind.A, self.Kind.B}', 'return k in {1, 2}'),         # right of `in`
    ('return T.loc[(self.Kind.A, k), "q"]', 'T.loc[(1, k), "q"]'),            # a subscript key
])
def test_container_displays(body, expected):
    assert expected in translate(body)


@pytest.mark.parametrize('body', [
    'return self.Loose.X',                      # not enum-safe
    'return self.rate',                         # not an enum
    'return self.Other.A',                      # a Reference of another class
    'return self.Kind.MISSING',                 # not a member
    'self = k\nreturn self.Kind.A',             # `self` rebound
    'def g(self):\n    return self.Kind.A\nreturn g',     # a nested function with its own `self`
])
def test_other_attributes_stay(body):
    assert translate(body) == f'class _c_P:\n    def f(self, k):\n        ' + body.replace('\n', '\n        ') + '\n'


CELLS = {'m._c_P.f': SimpleNamespace(arg_types={'k': Kind}, ret_type=SimpleNamespace(value_type=float, is_array=False)),
         'm._c_P._f_kind': SimpleNamespace(arg_types={}, ret_type=SimpleNamespace(value_type=Kind, is_array=False))}


def named(body):
    source = 'class _c_P:\n    def f(self, k):\n        ' + body.replace('\n', '\n        ') + '\n'
    return literals('m', source, REFS, CELLS)


@pytest.mark.parametrize('body, expected', [
    ('return self.Kind.A.name', "return 'A'"),                          # a member
    ('return self.Kind.B.value', 'return (2)'),
    ('return k.name', "return ({-3: 'LOW', 1: 'A', 2: 'B'}[k])"),      # a parameter traced as Kind
    ('return k.value + 1', 'return (k) + 1'),
    ('x = self.Kind.A if k else self.Kind.B\nreturn x.name', "return ({-3: 'LOW', 1: 'A', 2: 'B'}[x])"),   # a local
    ('for x in (self.Kind.A, self.Kind.B):\n    y = x.name', "y = ({-3: 'LOW', 1: 'A', 2: 'B'}[x])"),
    ('return self.kind(t).name', "{-3: 'LOW', 1: 'A', 2: 'B'}[self.kind(t)]"),   # a Cell result
    ('return T.loc[(k.name, 3), "q"]', "T.loc[(({-3: 'LOW', 1: 'A', 2: 'B'}[k]), 3), \"q\"]"),  # a table key
])
def test_name_and_value_of_enum_values(body, expected):
    assert expected in named(body)


@pytest.mark.parametrize('body', [
    'x = self.Kind.A\nx = 3\nreturn x.name',          # a local also bound to an int
    'k = 2\nreturn k.name',                            # a parameter rebound
    'return [k.name for k in names]',                  # shadowed by a comprehension
])
def test_other_names_stay(body):
    assert '.name' in named(body)


@pytest.mark.parametrize('body', ['return k.bit_length()', 'return self.Kind.A.real', 'return self.kind(t).label'])
def test_other_attributes_of_enum_values_are_build_errors(body):
    with pytest.raises(TypeError, match=r'm\._c_P\.f: .* reads \.\w+ of an IntEnum value'):
        named(body)


def test_no_enum_reference_leaves_the_source_unparsed():
    source = 'class _c_P:\n    def f(self):  return  self.Kind.A\n'
    assert literals('n', source, REFS) is source


@pytest.mark.parametrize('body', ['return k is self.Kind.A', 'return self.Kind.B is not k',
                                  'return 0 < k is self.Kind.A'])
def test_identity_with_a_member_is_a_build_error(body):
    with pytest.raises(TypeError, match='by identity'):
        translate(body)
