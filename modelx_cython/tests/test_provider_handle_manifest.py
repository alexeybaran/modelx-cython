"""M1 (W1.1): the optional ``handle`` declaration of a provider type."""

import pytest

from modelx_cython.provider_manifest import TypedHandle, _read_type, read_provider_manifest

TABLE = 'sample_provider.Table'
RECORD = 'sample_provider.Record'


def test_handle_is_optional_and_defaults_to_none():
    assert _read_type('pkg.mod.Plain', {}).handle is None


@pytest.mark.parametrize('handle', [
    {'type': 'TableHandle'},                                     # build missing
    {'type': 'TableHandle', 'build': 'table_handle', 'x': 'y'},  # unknown field
    {'type': 'Table Handle', 'build': 'table_handle'},           # not an identifier
    'TableHandle',
])
def test_invalid_handle_declarations_are_rejected(handle):
    with pytest.raises(ValueError):
        _read_type('pkg.mod.Plain', {'handle': handle})
