"""Shared pytest configuration of the compiler's tests."""
import os

# Every translation the tests run (in this process or in an mx2cy subprocess) re-parses the
# text it writes and fails on a dropped parenthesis: the build itself no longer checks.
os.environ['MX2CY_REPARSE_CHECK'] = '1'
