"""One construction and one release of a top-level item.

A directly created item is ``C(base)`` with the base's References and itself as its only root, built
once per key. ``__delitem__`` releases the item it drops through one shared function. Releasing breaks the item's references to itself (``_space``,
``_mx_roots``) and its Cell objects, so it is freed by its reference count.
"""
import libcst as cst

from .consts import CY_MOD


def new_root(cls_name):
    return f'_mx_new_root_{cls_name}'


def release_root(cls_name):
    return f'_mx_release_root_{cls_name}'


def release_item(cls_name):
    return f'_mx_release_item_{cls_name}'


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
        f"    root._space = None\n"
        f"\n\n"
        f"@{CY_MOD}.cfunc\n"
        f"def {release_item(cls_name)}(removed: object) -> None:\n"
        f"    # An item dropped from its ItemSpace (``del``, whole-model clear): released as above when it\n"
        f"    # is a compiled item without child Spaces or items of its own; anything else is left as is.\n"
        f"    item: {cls_name}\n"
        f"    if type(removed) is {cls_name}:\n"
        f"        item = {CY_MOD}.cast({cls_name}, removed)\n"
        f"        if not item._mx_spaces and not item._mx_itemspaces:\n"
        f"            {release_root(cls_name)}(item)\n").body
