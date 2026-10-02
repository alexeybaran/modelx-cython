"""Model argument names cannot collide with generated checked bridge locals."""
import json

from test_original_frame_shell import _run


def test_checked_bridge_reserves_only_its_internal_names(tmp_path):
    import modelx as mx

    model = mx.new_model('CheckedNames')
    space = model.new_space('S')
    space.lookup = {1: 10.0, 2: -1.0}
    space.new_cells('checked', '''def checked(scope, owner, value):
    return float(lookup[scope]) + owner + value
''')
    model.export(tmp_path / 'model')
    model.close()
    (tmp_path / 'sample.py').write_text(
        'from model import mx_model\nmx_model.S.checked(1, 1, 1)\nmx_model.S.checked(2, 1, 1)\n',
        encoding='utf-8')
    _run(tmp_path, ['-m', 'modelx_cython', str(tmp_path / 'model'), '--no-spec',
                    '--sample', str(tmp_path / 'sample.py')], 'build')
    manifest = json.loads((tmp_path / 'model_cy/_mx_frame_S/manifest.json').read_text())
    assert 'checked' in manifest['original_checked_numeric_bridges']
    (tmp_path / 'verify.py').write_text('''
from model_cy import mx_model
assert mx_model.S.checked(1, 1, 1) == 12.0
assert mx_model.S.checked(2, 1, 1) == 1.0
mx_model.S.lookup = {1: 100.0, 2: 200.0}
assert mx_model.S.checked(1, 1, 1) == 12.0
assert mx_model.S.checked(2, 0, 0) == 200.0
''', encoding='utf-8')
    _run(tmp_path, ['verify.py'], 'verify')
