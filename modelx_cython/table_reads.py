"""Type-directed lowering of pandas-spelled table reads to typed handle reads (M1).

W1.1 stub (D26): the contract the transformer hook calls; W1.3 owns this module and fills
``lower``. The stub lowers nothing, so every read found by
``admission.table_read_sites`` is reported untyped and the formula is returned unchanged
(the same node object, so generated code is byte-identical).

Contract (fixed by W1.1, see round3/design/M1-INTERFACE.md):

- ``TableReads(resolver, cls_info, cells, providers, policy, sites)``; ``sites`` is the list
  of ``admission.TableReadSite`` of the formula, found on the node passed to ``lower``.
- ``lower(node) -> node``: the formula ``FunctionDef`` with the reads it can type rewritten
  to handle calls (``_native_inputs.TableHandle``); return ``node`` itself when nothing changes.
- after ``lower``: ``typed`` (int, reads emitted as handle reads), ``lowered`` (set of
  ``id(site.node)`` of the sites it lowered), ``declined`` (dict ``id(site.node)`` -> reason
  for sites it saw but kept), ``keys`` (list of str notes on the key parts of typed reads).
"""


class TableReads:
    def __init__(self, resolver, cls_info, cells, providers, policy, sites):
        self.resolver, self.cls_info, self.cells = resolver, cls_info, cells
        self.providers, self.policy, self.sites = providers, policy, sites
        self.typed = 0
        self.lowered = set()
        self.declined = {}
        self.keys = []

    def lower(self, node):
        return node
