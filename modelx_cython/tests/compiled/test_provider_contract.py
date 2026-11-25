import pytest

from modelx_cython.tracer import RuntimeValueInfo


def test_helper_metadata_rejects_invalid_capability():
    def value():
        return 0

    value.__modelx_cython_helper__ = ("sample.provider", "lookup")
    value.__modelx_cython_scalar_result__ = "complex"
    with pytest.raises(ValueError, match="invalid helper capability 'scalar_result'"):
        RuntimeValueInfo(value)
