"""Local result containers remain ordinary compiled Cython variables."""
import json

from test_original_frame_shell import _run


def test_local_names_matching_unrelated_cells_remain_compiled(tmp_path):
    import modelx as mx

    model = mx.new_model('LocalNames')
    space = model.new_space('S')
    space.new_cells('base', 'def base():\n return 10.0\n')
    space.new_cells('value', 'def value(n):\n return float(n)\n')
    space.new_cells('results', '''def results():
    base = []
    for n in range(3):
        base.append(value(n))
    return {'base': base}
''')
    model.export(tmp_path/'model')
    model.close()
    classes = tmp_path/'model/_mx_classes.py'
    text = classes.read_text()
    text = text.replace('base.append(self.value(n))', 'base.append(self.value(n) + self.base())')
    classes.write_text(text)
    (tmp_path/'model/_mx_formula_sources.json').unlink(missing_ok=True)
    (tmp_path/'sample.py').write_text(
        'from model import mx_model\nmx_model.S.base();mx_model.S.results()\n',
        encoding='utf-8')
    _run(tmp_path, ['-m','modelx_cython',str(tmp_path/'model'),'--no-spec',
                   '--sample',str(tmp_path/'sample.py')], 'build')
    manifest = json.loads((tmp_path/'model_cy/_mx_frame_S/manifest.json').read_text())
    assert manifest['cells']['results']['coverage']['body_backend'] == 'cython_gil'
    assert not manifest['cells']['results']['coverage']['detached_python']
    (tmp_path/'verify.py').write_text('''
from model_cy import mx_model
result = mx_model.S.results()
assert result == {'base': [10.0, 11.0, 12.0]}
assert mx_model.S.results() is result
assert mx_model.S.base() == 10.0
''', encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')


def test_real_first_class_cell_uses_original_checked_wrapper(tmp_path):
    import modelx as mx

    model = mx.new_model('FirstClassCell')
    space = model.new_space('S')
    space.new_cells('value', 'def value(n):\n return float(n + 1)\n')
    space.new_cells('expose', 'def expose():\n return value\n')
    space.new_cells('consume', 'def consume(n):\n return expose()(n)\n')
    model.export(tmp_path/'model')
    model.close()
    (tmp_path/'sample.py').write_text(
        'from model import mx_model\nmx_model.S.consume(1);mx_model.S.consume(2)\n',
        encoding='utf-8')
    _run(tmp_path, ['-m','modelx_cython',str(tmp_path/'model'),'--no-spec',
                   '--sample',str(tmp_path/'sample.py')], 'build')
    routes = json.loads((tmp_path/'model_cy/_mx_classes.frame-shell.json').read_text())
    expose = next(r for r in routes['cells'] if r['cell']=='expose')
    assert expose['execution'] == 'original_cython'
    assert 'first-class Cell' in expose['reason']
    manifest = json.loads((tmp_path/'model_cy/_mx_frame_S/manifest.json').read_text())
    assert manifest['cells']['consume']['coverage']['body_backend'] == 'cython_gil'
    assert 'expose' in manifest['inputs']
    (tmp_path/'verify.py').write_text('from model_cy import mx_model\ncallback = mx_model.S.expose()\nassert callback(1) == mx_model.S.consume(1) == 2.0\nassert callback(2) == mx_model.S.consume(2) == 3.0\nassert mx_model.S.expose() is callback\n', encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')


def test_qualified_cell_and_public_parameter_keep_distinct_bindings(tmp_path):
    import modelx as mx
    model = mx.new_model('ParameterNames')
    space = model.new_space('S')
    space.new_cells('leaf', 'def leaf():\n return 2.0\n')
    space.new_cells('total', 'def total(leaf):\n return leaf + 2.0\n')
    space.new_cells('parent', 'def parent(n):\n return total(n)\n')
    model.export(tmp_path/'model')
    model.close()
    classes = tmp_path/'model/_mx_classes.py'
    text = classes.read_text().replace('return leaf + 2.0', 'return leaf + self.leaf()')
    classes.write_text(text)
    (tmp_path/'model/_mx_formula_sources.json').unlink(missing_ok=True)
    (tmp_path/'sample.py').write_text('from model import mx_model\nmx_model.S.parent(3.0)\n')
    _run(tmp_path, ['-m','modelx_cython',str(tmp_path/'model'),'--no-spec',
                   '--sample',str(tmp_path/'sample.py')], 'build')
    rows = json.loads((tmp_path/'model_cy/_mx_classes.frame-shell.json').read_text())['cells']
    total = next(row for row in rows if row['cell']=='total')
    assert total['execution'] == 'original_cython'
    assert 'public argument' in total['reason']
    manifest = json.loads((tmp_path/'model_cy/_mx_frame_S/manifest.json').read_text())
    assert 'total' in manifest['inputs']
    assert 'parent' in manifest['cells']
    (tmp_path/'probe.pyx').write_text('''from model_cy._mx_classes cimport _c_S
cpdef double private_total(_c_S space, double argument):
    return space._f_total(argument)
''')
    (tmp_path/'probe_setup.py').write_text('''from setuptools import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize('probe.pyx', language_level=3))
''')
    _run(tmp_path, ['probe_setup.py','build_ext','--inplace'], 'probe-build')
    (tmp_path/'verify.py').write_text('''from model_cy import mx_model
from probe import private_total
assert mx_model.S.total(3.0) == mx_model.S.parent(3.0) == 5.0
assert mx_model.S.total(leaf=4.0) == 6.0
assert private_total(mx_model.S, 5.0) == 7.0
''')
    _run(tmp_path, ['verify.py'], 'verify')
