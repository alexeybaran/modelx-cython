"""Build a one-Space model from formulas and compare every cell, live against compiled."""
import json
import pickle
import subprocess
import sys
import textwrap

import modelx as mx

PROBE = textwrap.dedent('''\
    import pickle, sys
    from importlib import import_module
    space = import_module(sys.argv[1]).mx_model.Projection
    names, arguments = pickle.loads(sys.stdin.buffer.read())
    result = {}
    for name in names:
        for argument in arguments:
            try:
                value = getattr(space, name)(argument)
            except Exception as error:
                result[name, argument] = ('raise', type(error).__name__, str(error))
            else:
                result[name, argument] = (type(value).__name__, repr(value))
    sys.stdout.buffer.write(pickle.dumps(result))
    ''')


def outcomes(tmp_path, cells, checked, arguments, refs=None, name='ReductionCases'):
    """{package: {(cell, argument): (type name, repr) or ('raise', type name, text)}} for the exported
    model and its compiled build. ``cells`` maps names to formulas; the ``checked`` ones are compared."""
    model = mx.new_model(name)
    try:
        space = model.new_space('Projection', refs=refs or {})
        for cell, formula in cells.items():
            space.new_cells(cell, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(textwrap.dedent(f'''\
        from exported import mx_model as m
        for name in {list(cells)!r}:
            for argument in {list(arguments)!r}:
                try: getattr(m.Projection, name)(argument)
                except Exception: pass
        '''), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)
    request = pickle.dumps((list(checked), list(arguments)))
    return {package: pickle.loads(subprocess.run(
        [sys.executable, '-c', PROBE, package], cwd=tmp_path, check=True, input=request,
        capture_output=True).stdout) for package in ('exported', 'exported_cy')}


def assert_equal_live(tmp_path, cells, checked, arguments, **options):
    """Type, repr and exception equal. One stated exception: a cell whose compiled return type is double
    converts an int result at its boundary (live ``0``, compiled ``0.0``), which no reduction changes."""
    result = outcomes(tmp_path, cells, checked, arguments, **options)
    live, compiled = result['exported'], result['exported_cy']
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    returns = {name: report[f'exported_cy._mx_classes._c_Projection._f_{name}']['return_type'] for name in checked}
    differences = {}
    for key in live:
        boundary = (returns[key[0]] == 'double' and live[key][0] == 'int'
                    and compiled[key] == ('float', repr(float(int(live[key][1])))))
        if live[key] != compiled[key] and not boundary:
            differences[key] = (live[key], compiled[key])
    assert differences == {}, differences
    return live
