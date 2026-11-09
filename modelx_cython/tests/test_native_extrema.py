import pytest

from modelx_cython.native_extrema import capture_extrema_inputs


def test_build_hook_is_scoped_even_if_cythonization_fails():
    from Cython.Compiler.Optimize import EarlyReplaceBuiltinCalls
    original = EarlyReplaceBuiltinCalls._optimise_min_max
    with pytest.raises(RuntimeError, match='build failed'):
        with capture_extrema_inputs():
            assert EarlyReplaceBuiltinCalls._optimise_min_max is not original
            raise RuntimeError('build failed')
    assert EarlyReplaceBuiltinCalls._optimise_min_max is original
