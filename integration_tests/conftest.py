"""Shared pytest configuration of the integration tests."""
import os

# As in modelx_cython/tests: every translation re-parses the text it writes (reparse_check).
os.environ['MX2CY_REPARSE_CHECK'] = '1'
