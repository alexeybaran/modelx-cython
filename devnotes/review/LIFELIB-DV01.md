# SEC DV01 and CS01 example on the MYGA model

The Direct example builds a key-rate DV01 and CS01 report using MYGA SEC cash flows and G2++ scenarios. It accepts an external CSV directory, with the policy book alongside the other model-owned input tables. The launcher checks its build stamp before reusing a compiled package. The Direct example needs the dynamic Space Reference exporter.

The Collect follow-up declares the Actual opening fields and Valuation cash flows it needs. Both routes keep the same CSV interface and use the same valuation formulas. The compiled report can use one or several workers. Collect preserves the shared model inputs and releases policy calculation state as it gathers the requested outputs.

Both review heads are based on lifelib main `fbafbaab44493f77bdd015ad09f0d770a6488ded`. The authored DV01 files are byte-identical to the earlier examples. The two changed files of the upstream G2++ source have the same executable syntax trees after removing docstrings. On the rebased sources, the 12 Direct live tests and 13 mixed live tests passed, including the 72 report values and both stored totals. The earlier compiled full-report qualification and scaling results keep their original lifelib pins.

Review the Direct example after the exporter and compiler it needs. The Collect follow-up depends on the Collect modelx and compiler proposals. [Measured full report and scalar checks](https://github.com/alexeybaran/modelx-cython/blob/review/fast-sec-collect-evidence/devnotes/review/PERFORMANCE.md)
