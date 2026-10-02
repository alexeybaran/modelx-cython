"""Untraced formulas use original object signatures on the shared frame path."""
import json

from test_original_frame_shell import _run


def test_untraced_object_cells_preserve_cache_and_private_abi(tmp_path):
    import modelx as mx

    model = mx.new_model('UntracedShell')
    space = model.new_space('S')
    space.offset = 2.0
    space.fail = False
    space.counts = {}
    for name, source in {
        'observed': 'def observed(t):\n return float(t) + offset\n',
        'unseen': ('def unseen(key):\n counts[key] = counts.get(key, 0) + 1\n'
                   ' value = observed(0)\n'
                   ' if fail: raise ValueError("unseen failure")\n'
                   ' return {"key": key, "value": value}\n'),
        'nullable': 'def nullable(key):\n return None if key is None else unseen(key)\n',
        'uncached': 'def uncached(key):\n return {"value": unseen(key)["value"] + offset}\n',
    }.items():
        space.new_cells(name=name, formula=source)
    space.nullable.allow_none = True
    space.uncached.is_cached = False
    model.export(tmp_path / 'model')
    model.close()
    metadata = tmp_path / 'model/_mx_formula_sources.json'
    if metadata.exists():
        metadata.unlink()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\nmx_model.S.observed(0)\n', encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--no-spec',
                    '--sample', str(tmp_path / 'sample.py')], 'build')
    plan = json.loads((tmp_path / 'model_cy/_mx_compilation_plan.json').read_text())
    cells = plan['spaces']['model_cy.S']['cells']
    assert cells['unseen']['traced'] is False
    assert cells['unseen']['public_result'] == 'object'
    assert cells['unseen']['public_arguments'] == {'key': 'object'}
    routes = json.loads((tmp_path / 'model_cy/_mx_classes.frame-shell.json').read_text())
    assert {row['cell'] for row in routes['cells'] if row['execution'] == 'frame_exact'} == set(cells)
    (tmp_path / 'override.pyx').write_text('''
from model_cy._mx_classes cimport _c_S
cdef class Private(_c_S):
    cdef object _f_unseen(self, object key):
        return {'key': key, 'value': 40.0}
def private_unseen(_c_S space, object key):
    return space._f_unseen(key)
def frame_owner(_c_S space):
    return space._mx_frame_owner
''', encoding='utf-8')
    (tmp_path / 'override_setup.py').write_text(
        "from setuptools import setup\nfrom Cython.Build import cythonize\n"
        "setup(ext_modules=cythonize('override.pyx'))\n", encoding='utf-8')
    _run(tmp_path, ['override_setup.py', 'build_ext', '--inplace'], 'override')
    (tmp_path / 'verify.py').write_text('''
import pickle
from model_cy import mx_model
from override import Private, private_unseen, frame_owner
model = type(mx_model)(); space = model.S
assert space.nullable(None) is None
key = ('object', 1)
first = space.unseen(key)
assert first == {'key': key, 'value': 2.0}
assert space.unseen(key) is first and space.counts[key] == 1
space.offset = 5.0
assert space.unseen(key) is first
assert space.uncached(key) == {'value': 7.0}
space.offset = 8.0
assert space.uncached(key) == {'value': 10.0}
# Private formula execution remains uncached and shares completed child results.
assert private_unseen(space, key) == first and space.counts[key] == 2
assert space.unseen(key) is first
space.fail = True
for _ in range(2):
    try: space.unseen('retry')
    except ValueError as error: assert str(error) == 'unseen failure'
    else: raise AssertionError('failed untraced Cell cached')
assert space.counts['retry'] == 2
space.fail = False
assert space.unseen('retry')['value'] == 2.0
assert frame_owner(space).kernel.call('unseen', (key,)) is first
restored = pickle.loads(pickle.dumps(space))
restored.offset = 100.0
assert restored.unseen(key) == first
assert restored.nullable(None) is None
derived = Private(model); derived.offset = 9.0; derived.fail = False; derived.counts = {}
assert derived.nullable(key) == {'key': key, 'value': 40.0}
assert frame_owner(derived).kernel.call('unseen', (key,)) == derived.unseen(key)
print('untraced compiled objects and original private ABI: PASS')
''', encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')
