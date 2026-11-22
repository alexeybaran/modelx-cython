"""Check the reconciled lifelib input and cached-list contracts."""

from pathlib import Path
from tempfile import TemporaryDirectory

import modelx as mx
import pandas as pd


def verify_regional_contract(root):
    products = root / 'lifelib/libraries'
    outcomes = {}
    fixtures = {
        'Medical_KR_S': {'mort_table': 'mort_table_file',
                         'utilisation_table': 'utilisation_table_file',
                         'severity_table': 'severity_table_file'},
        'MYGA_US_S': {'surr_charge_table': 'surr_charge_file'},
    }
    for name, readers in fixtures.items():
        source, = products.glob(f'*lib/products/*/{name}')
        model = mx.read_model(source)
        try:
            with TemporaryDirectory() as directory:
                for cell, reference in readers.items():
                    data = model.Data
                    original = getattr(data, cell)()
                    filename = source.parent / getattr(data, reference)
                    replacement = Path(directory) / filename.name
                    replacement.write_bytes(filename.read_bytes())
                    setattr(data, reference, replacement)
                    reloaded = getattr(data, cell)()
                    pd.testing.assert_frame_equal(original, reloaded)
                    outcomes[f'{name}.{cell}'] = {'csv': filename.name, 'rows': len(reloaded)}
        finally:
            model.close()

    source, = products.glob('*lib/products/*/IP_UK_S')
    model = mx.read_model(source)
    try:
        claim = model.Projection[5]
        rates = claim.claim_rate_vectors()
        cohorts = claim.sick_cohorts(2)
        baseline = cohorts[:]
        assert rates and all(type(rate) is list for rate in rates)
        assert type(cohorts) is list and type(claim.sick_cohorts(3)) is list
        assert cohorts == baseline and claim.sick_cohorts(2) is cohorts
        outcomes['IP_UK_S.cached_lists'] = {'rate_vectors': len(rates), 'cached_cohorts': len(cohorts)}
    finally:
        model.close()
    return outcomes
