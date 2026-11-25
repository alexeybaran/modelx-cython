# The cache sizes of MYGA_US_SEC's spec_sec.py, for Projection and Valuation (its formulas); Market's
# vectors by month t, scenario s: (1322, 8), the one inner width of every 2-D cache of the model.
# Closes up to 15 (the runner's per-thread Market experiment uses closes 3k..3k+2).
{"spaces": {
    "Projection": {"cells_param_size": {
        "t": 1322, ("t", "timing"): (1322, 8), ("t", "kind"): (1322, 8),
        ("t", "name"): (1322, 8), "y": 102, "a": 101}},
    "Valuation": {"cells_param_size": {
        "t": 1322, ("t", "timing"): (1322, 8), ("t", "kind"): (1322, 8),
        ("t", "name"): (1322, 8), "y": 102, "a": 101}},
    "Market": {"cells_param_size": {"t": 1322, ("t", "s"): (1322, 8)}},
    "ESG": {"cells_param_size": {"close": 16}},
    "Outer": {"cells_param_size": {"close": 16, ("close", "shock"): (16, 8)}}}}
