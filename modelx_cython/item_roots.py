"""D116: one construction and one release of a top-level item, shared by collect and the direct route.

Collect's reused root and a directly created item are the same thing: ``C(base)`` with the base's
References and itself as its only root. Collect builds one per call and resets it per policy; the
direct route builds one per key. Releasing breaks the item's references to itself (``_space``,
``_mx_roots``) and its Cell objects, so it is freed by its reference count.
"""
import libcst as cst

from .consts import CY_MOD


def new_root(cls_name):
    return f'_mx_new_root_{cls_name}'


def release_root(cls_name):
    return f'_mx_release_root_{cls_name}'


def functions(cls_name):
    """Module-level C functions for one compiled ItemSpace class."""
    return cst.parse_module(
        f"@{CY_MOD}.cfunc\n"
        f"def {new_root(cls_name)}(base: {cls_name}) -> {cls_name}:\n"
        f"    root: {cls_name} = {cls_name}(base)\n"
        f"    root._mx_copy_refs(base, base)\n"
        f"    root._mx_roots.append(root)\n"
        f"    return root\n"
        f"\n\n"
        f"@{CY_MOD}.cfunc\n"
        f"def {release_root(cls_name)}(root: {cls_name}) -> None:\n"
        f"    root._mx_cells.clear()\n"
        f"    root._mx_roots.clear()\n"
        f"    root._space = None\n").body
