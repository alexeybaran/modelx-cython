"""Canonical public Enum identity in the experimental frame backend."""
from enum import IntEnum

import pytest

from modelx_cython._frame_runtime.compiler import type_info
from modelx_cython._frame_runtime.contracts import ContractError
from modelx_cython._frame_runtime.enum_binding import resolve_enum
from modelx_cython._frame_runtime.python_calculator import bundle_source


class PublicKind(IntEnum):
    FIRST = 0
    SECOND = 1


def test_importable_enum_keeps_exact_public_class():
    info = type_info(PublicKind)
    assert info['module'] == PublicKind.__module__
    assert resolve_enum(info) is PublicKind
    source, _, _ = bundle_source({}, (), {}, {'PublicKind': PublicKind})
    namespace = {}
    exec(source, namespace)
    assert namespace['PublicKind'] is PublicKind


def test_changed_public_enum_domain_is_rejected():
    info = type_info(PublicKind)
    info['members'] = {'FIRST': 0, 'SECOND': 2}
    with pytest.raises(ContractError, match='Public Enum contract changed'):
        resolve_enum(info)
