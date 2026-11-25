"""Lexical analysis of exported modules: the storage of the exported collection."""
import textwrap

from modelx_cython.parser import ModuleVisitor


def test_exported_collection_and_dynamic_ref_storage():
    source = textwrap.dedent('''\
        class _c_Actual:
            __slots__ = ("__dict__", "_mx_collect_cache")

            def __init__(self, parent):
                self._mx_collect_cache = {}

        class _c_Static:
            __slots__ = ("name",)
        ''')
    visitor = ModuleVisitor(module="M._mx_classes", source=source)

    assert visitor.collect_cache_classes == {"_c_Actual"}
    assert visitor.dynamic_ref_classes == {"_c_Actual"}
    assert visitor.spaces == {}
