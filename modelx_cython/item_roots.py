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


def window_lines(cls, receiver='root'):
    """D74/D123: the dense windows of the class's hash-cached Cells, set once on every new root.

    The same storage for the collect root and for every item (D116): collect differs only in
    reusing its root and resetting what was written between policies.
    """
    lines = []
    for name, (axes, _, _) in sorted(getattr(cls, 'reuse_dense_windows', {}).items()):
        bounds = [f'{int(a.lower)}, {int(a.stop) - int(a.lower)}' for a in axes]
        bounds += ['0, 1'] * (3 - len(bounds))
        # D126: geometry only (storage on the first miss); the written list only on a reset receiver
        lines.append(f'{receiver}._v_{name}.window({", ".join(bounds)}, track)')
    return lines


def functions(cls_name, cls=None):
    """Module-level C functions for one compiled ItemSpace class."""
    windows = ''.join(f'    {line}\n' for line in window_lines(cls)) if cls is not None else ''
    return cst.parse_module(
        f"@{CY_MOD}.cfunc\n"
        f"def {new_root(cls_name)}(base: {cls_name}, track: {CY_MOD}.bint) -> {cls_name}:\n"
        f"    root: {cls_name} = {cls_name}(base)\n"
        + windows +
        f"    root._mx_copy_refs(base, base)\n"
        f"    root._mx_roots.append(root)\n"
        f"    return root\n"
        f"\n\n"
        f"@{CY_MOD}.cfunc\n"
        f"def {release_root(cls_name)}(root: {cls_name}) -> None:\n"
        f"    root._mx_cells.clear()\n"
        f"    root._mx_roots.clear()\n"
        f"    root._space = None\n").body
