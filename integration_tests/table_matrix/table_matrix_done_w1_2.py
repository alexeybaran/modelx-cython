"""Squares done by W1.2 (runtime handles): {(form, operation, key, mode)}. Owned by W1.2.

Every applicable square, in every mode: the compiled reads call the typed handles directly
(table_matrix_w1_2_driver.pyx) in the shape W1.3's lowering emits.
"""
from table_matrix_cases_w1_2 import CASES
from table_matrix_squares import MODES

DONE = {(form, operation, key, mode) for form, operation, key in CASES for mode in MODES}
