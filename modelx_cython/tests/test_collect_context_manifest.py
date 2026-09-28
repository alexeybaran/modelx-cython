"""Collect context metadata includes object edges without model exclusion."""

import hashlib

import pytest

from modelx_cython.collect_context_manifest import build_manifest


EXPORT = '''
class _c_Projection:
    def _f_collect_outputs(self):
        return {'sum': {'total': self.total(self.category())}}
    def _f_total(self, key):
        return self.value() + key
    def _f_value(self):
        return 3.0
    def _f_category(self):
        return 'base'
    def _f_unrelated(self):
        return 4.0
'''

COMPILED = '''
class _c_Projection:
    def _f_collect_outputs(self) -> object: pass
    def _f_total(self, key: object) -> _mx_cy.double: pass
    def _f_value(self) -> _mx_cy.double: pass
    def _f_category(self) -> str: pass
    def _f_unrelated(self) -> _mx_cy.double: pass
'''


def test_manifest_keeps_object_dependency_and_pins_sources(tmp_path):
    export = tmp_path / 'export.py'
    compiled = tmp_path / 'compiled.py'
    export.write_text(EXPORT, encoding='utf-8')
    compiled.write_text(COMPILED, encoding='utf-8')
    manifest = build_manifest(export, compiled)
    assert manifest['roots'] == ['category', 'total']
    assert set(manifest['cells']) == {'category', 'total', 'value'}
    assert manifest['numeric'] == ['value']
    assert manifest['object_edges'] == ['category', 'total']
    assert manifest['cells']['total']['arguments'] == {'key': 'object'}
    assert manifest['cells']['total']['dependencies'] == ['value']
    assert manifest['export_sha256'] == hashlib.sha256(export.read_bytes()).hexdigest()
    assert manifest['cython_sha256'] == hashlib.sha256(compiled.read_bytes()).hexdigest()


def test_missing_compiled_signature_blocks_manifest(tmp_path):
    export = tmp_path / 'export.py'
    compiled = tmp_path / 'compiled.py'
    export.write_text(EXPORT, encoding='utf-8')
    compiled.write_text(COMPILED.replace('def _f_value', 'def _f_other'), encoding='utf-8')
    with pytest.raises(ValueError, match='missing compiled Cell signature: value'):
        build_manifest(export, compiled)


def test_changed_argument_order_blocks_manifest(tmp_path):
    export = tmp_path / 'export.py'
    compiled = tmp_path / 'compiled.py'
    export.write_text(EXPORT, encoding='utf-8')
    compiled.write_text(COMPILED.replace('_f_total(self, key: object)',
                                         '_f_total(self, renamed: object)'),
                        encoding='utf-8')
    with pytest.raises(ValueError, match='compiled Cell arguments differ: total'):
        build_manifest(export, compiled)
