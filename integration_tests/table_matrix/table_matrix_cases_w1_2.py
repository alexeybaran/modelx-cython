"""Square cases contributed by W1.2: {(form, operation, key): case}. Owned by W1.2.

A case is ``case(mode, tmp_path_factory)`` that runs the square in ``mode`` ('live',
'exported' or 'compiled') and asserts the pandas result (oracle) and, in compiled, that no
Python fallback ran. Share one build per form (7 builds) through a session fixture.
"""
CASES = {}
