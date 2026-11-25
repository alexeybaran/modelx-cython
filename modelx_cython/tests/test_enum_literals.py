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
    ('return k.name', "return ({-3: 'LOW', 1: 'A', 2: 'B'}.get(k) or getattr(k, 'name'))"),   # a parameter
    ('return k.value + 1', 'return (k) + 1'),
    ('x = self.Kind.A if k else self.Kind.B\nreturn x.name',                                       # a local
     "return ({-3: 'LOW', 1: 'A', 2: 'B'}.get(x) or getattr(x, 'name'))"),
    ('for x in (self.Kind.A, self.Kind.B):\n    y = x.name',
     "y = ({-3: 'LOW', 1: 'A', 2: 'B'}.get(x) or getattr(x, 'name'))"),
    ('return self.kind(t).name', "{-3: 'LOW', 1: 'A', 2: 'B'}.get(self.kind(t))"),   # a Cell result
    ('return T.loc[(k.name, 3), "q"]',                                                              # a table key
     "T.loc[(({-3: 'LOW', 1: 'A', 2: 'B'}.get(k) or getattr(k, 'name')), 3), \"q\"]"),
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
    with pytest.raises(TypeError, match=r'^P\.f: .* reads \.\w+ of an IntEnum value'):
        named(body)


def test_no_enum_reference_leaves_the_source_unparsed():
    source = 'class _c_P:\n    def f(self):  return  self.Kind.A\n'
    assert literals('n', source, REFS) is source


@pytest.mark.parametrize('body', ['return k is self.Kind.A', 'return self.Kind.B is not k',
                                  'return 0 < k is self.Kind.A'])
def test_identity_with_a_member_is_a_build_error(body):
    with pytest.raises(TypeError, match='by identity'):
        translate(body)


# ---------------------------------------------------------------- review fixes: results equal live or a build error
class Leg(enum.IntEnum):
    CALL = 1
    PUT = 2


class Pretty(enum.IntEnum):         # its own str and format: not enum-safe, but .name of its values works
    M = 0
    F = 1

    def __str__(self):
        return self.name

    __format__ = lambda self, spec: self.name


REVIEW_REFS = {**REFS, 'm._c_P.Leg': info(Leg)}
REVIEW_CELLS = {**CELLS, 'm._c_P._f_pretty': SimpleNamespace(arg_types={'s': Pretty}, ret_type=None),
                'm._c_P._f_leg': SimpleNamespace(
                    arg_types={}, ret_type=SimpleNamespace(value_type=Leg, is_array=False)),
                'm._m_P._mx_classes._c_O._f_pick': SimpleNamespace(
                    arg_types={}, ret_type=SimpleNamespace(value_type=Kind, is_array=False))}


def reviewed(body, params='self, k', method='f'):
    source = f'class _c_P:\n    def {method}({params}):\n        ' + body.replace('\n', '\n        ') + '\n'
    return literals('m._mx_classes', source, {'m._mx_classes' + k[1:]: v for k, v in REVIEW_REFS.items()},
                    {'m._mx_classes' + k[1:] if k.startswith('m._c_') else k: v for k, v in REVIEW_CELLS.items()})


def test_enum_safe_needs_int_str_format_repr_and_hash():
    assert info(Kind).enum_safe and not info(Pretty).enum_safe


@pytest.mark.parametrize('body', [
    'return repr(self.Kind.A)', 'return ascii(self.Kind.A)', 'return type(self.Kind.A).__name__',
    'return isinstance(self.Kind.A, self.Kind)', 'return getattr(self.Kind.B, "name")', 'return id(self.Kind.A)',
    'return f"{self.Kind.A!r}"', 'return "%r" % self.Kind.A', 'return "%s %r" % (self.Kind.A, self.Kind.B)',
])
def test_a_member_whose_class_shows_stays_an_object(body):
    assert reviewed(body).endswith(body + '\n')


@pytest.mark.parametrize('body, expected', [
    ('return str(self.Kind.A)', 'return str(1)'), ('return f"{self.Kind.A}"', 'return f"{1}"'),
    ('return str(k) + format(k)', 'return str(k) + format(k)'),         # int's str and format: as live
    ('return "%d" % k', 'return "%d" % k'),
])
def test_str_and_format_of_an_enum_safe_value_stay_integers(body, expected):
    assert expected in reviewed(body)


@pytest.mark.parametrize('body, params, method', [
    ('return repr(k)', 'self, k', 'f'), ('return type(k).__name__', 'self, k', 'f'),
    ('return isinstance(self.kind(), int)', 'self, k', 'f'), ('return getattr(k, "name")', 'self, k', 'f'),
    ('return f"{k!r}"', 'self, k', 'f'), ('return "%r" % k', 'self, k', 'f'),
    ('return "%s %r" % (k, k)', 'self, k', 'f'),
    ('return "sex=" + str(s)', 'self, s', '_f_pretty'), ('return f"{s}"', 'self, s', '_f_pretty'),
])
def test_an_enum_value_held_as_an_integer_where_its_class_shows_is_a_build_error(body, params, method):
    with pytest.raises(TypeError, match=r'^P\.(f|pretty): .* the result would differ from live modelx'):
        reviewed(body, params, method)


@pytest.mark.parametrize('body', ['return self.f(self.Leg.PUT)', 'return self.f(k=self.Leg.CALL)',
                                  'return self.f(self.leg())'])
def test_a_member_of_another_class_given_to_an_enum_parameter_is_a_build_error(body):
    with pytest.raises(TypeError, match=r'^P\.f: .* traced as another IntEnum class'):
        reviewed(body)


def test_a_member_of_the_traced_class_given_to_an_enum_parameter_stays():
    assert 'return self.f(2)' in reviewed('return self.f(self.Kind.B)')


@pytest.mark.parametrize('body, expected', [
    ('return k.name', "return ({-3: 'LOW', 1: 'A', 2: 'B'}.get(k) or getattr(k, 'name'))"),
    ('return self.kind().name', "({-3: 'LOW', 1: 'A', 2: 'B'}.get(self.kind()) or getattr(self.kind(), 'name'))"),
])
def test_name_of_a_value_that_is_not_a_member_raises_as_live(body, expected):
    assert expected in reviewed(body)


@pytest.mark.parametrize('body', [
    'return tuple(x for x in (self.Kind.A,))[0] is (self.Kind.A, self.Kind.B)[0]',   # any identity test in the module
    'return k is self.pick()', 'x = self.Kind.A\ny = self.Kind.A\nreturn x is y', 'return k is self.enums.Kind.A',
])
def test_identity_in_a_module_with_enum_members_is_a_build_error(body):
    with pytest.raises(TypeError, match=r'^P\.f: .* by identity'):
        reviewed(body)


@pytest.mark.parametrize('body', ['return k == self.Kind.A is not None', 'return k is None',
                                  'return type(t) is int', 'return (t > 0) is True', 'return t is self.Color.RED'])
def test_identity_with_none_bools_and_builtin_types_stays(body):
    assert reviewed(body).endswith(body.replace('self.Kind.A', '1') + '\n')


@pytest.mark.parametrize('body, expected', [
    ('d = {}\nd[self.Kind.A] = 1.0\nreturn [x.name for x in d]', 'd[self.Kind.A] = 1.0'),     # a stored key
    ('return self.O.pick().name', "{-3: 'LOW', 1: 'A', 2: 'B'}.get(self.O.pick())"),            # a child Space's Cell
    ('x = self.O.pick()\nreturn x.name', "{-3: 'LOW', 1: 'A', 2: 'B'}.get(x)"),
])
def test_other_shapes_of_names(body, expected):
    assert expected in reviewed(body)


def test_name_of_an_enum_value_held_in_a_display_is_a_build_error():
    with pytest.raises(TypeError, match=r'^P\.f: .* holds an IntEnum value as an integer'):
        reviewed('return [self.kind()][0].name')
