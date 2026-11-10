"""Generic provider registration carries prepared pandas lookup into Cython."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd
from modelx_helpers.pandas_inputs import table as numeric_table
from modelx_helpers.typed_inputs import lookup


def test_pandas_provider_public_cache_override_and_pickle(tmp_path):
    model = mx.new_model('PandasProvider')
    try:
        space = model.new_space('Projection', refs={
            'lookup': lookup,
            'rates': numeric_table(pd.DataFrame({'rate': [0.1, 0.2, 0.3]}, index=[0, 1, 2]))})
        space.new_cells('rate', 'lambda key: lookup(rates, key, "rate")')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\nfor key in (0, 1, 2): mx_model.Projection.rate(key)\n')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                            '--sample', 'sample.py', '--no-spec'], cwd=tmp_path,
                           capture_output=True, text=True, env=dict(os.environ))
    (tmp_path / 'build.log').write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    probe = textwrap.dedent('''
        import pickle
        from pathlib import Path
        import pandas as pd
        from modelx_helpers.pandas_inputs import table as numeric_table
        from exported_cy import mx_model as m
        from modelx_helpers.typed_inputs import lookup as reference_lookup
        import sys
        helper_calls = []
        def profile(frame, event, arg):
            if event == 'call' and frame.f_code is reference_lookup.__code__:
                helper_calls.append(frame.f_code.co_name)
        s = m.Projection
        sys.setprofile(profile)
        try:
            assert s.rate(0) == s.rate(0) == 0.1
        finally:
            sys.setprofile(None)
        assert helper_calls == [], 'admitted native lookup called Python helper'

        # Prepared input replacement affects uncached calls; published caches stay.
        s.rates = numeric_table(pd.DataFrame({'rate': [1.0, 2.0, 3.0]}, index=[0, 1, 2]))
        assert s.rate(0) == 0.1
        assert s.rate(1) == 2.0
        restored = pickle.loads(pickle.dumps(m))
        assert restored.Projection.rate(0) == 0.1
        assert restored.Projection.rate(1) == 2.0
        assert restored.Projection.rate(2) == 3.0
        calls = []
        sentinel = RuntimeError('current callable')
        def replacement(table, key, column):
            calls.append((key, column))
            raise sentinel
        s.lookup = replacement
        try:
            s.rate(2)
        except RuntimeError as error:
            assert error is sentinel
        else:
            raise AssertionError('helper replacement bypassed')
        assert calls == [(2, 'rate')]
        assert s.rate(1) == 2.0
        providers = list(Path('exported_cy').glob('_mx_helper_*.pyx'))
        assert len(providers) == 1
        assert 'numeric_resource_value' in providers[0].read_text()
        sources = list(Path('exported_cy').glob('_mx_helper_*.c'))
        assert sources and '__pyx_v_data' in sources[0].read_text()
        assert 'lookup_f64' in Path('exported_cy/_mx_classes.py').read_text()
    ''')
    validation = subprocess.run([sys.executable, '-c', probe], cwd=tmp_path,
                                 capture_output=True, text=True, env=dict(os.environ))
    (tmp_path / 'validation.log').write_text(validation.stdout + validation.stderr)
    assert validation.returncode == 0, validation.stdout + validation.stderr
