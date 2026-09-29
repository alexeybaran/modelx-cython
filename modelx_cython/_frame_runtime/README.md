# Experimental native frame backend

This package is an in-progress transfer of the retained `prototype_0.8.2a1`
emitter and runtime into modelx-cython. It is kept off the accepted compiler
branch. The source snapshot is
`C:\mxw-01a0df78\h\references\prototype_0.8.2a1\src\mxcf_kernel`;
`modelx_frontend_native_mode.py` comes from the retained H06 frontend patch.

The generated kernel has one formula implementation and one cache layout per
Cell. `CompiledKernel.run_scalar` traverses policies with a reusable working
context. `CompiledKernel.fork_item` gives a directly accessed ItemSpace an
independent context with the same generated Cell entries. The two routes
therefore differ in **ownership**, not in formula or cache implementation.

The opt-in `mx2cy --frame-collect --no-spec` route can read preserved formula
sources from a standard exported package and build a separate frame package.
It does not replace the normal `mx2cy` translation path. Its experimental
facade accepts one flat policy Space. Direct zero-argument numeric `sum` leaves
use the original scalar path. Other `collect_outputs` shapes use one generated
Cython object Cell per policy; its result is detached and folded immediately,
including time-indexed `sum` and `individual` sections. The policy table may
also provide object-valued references prepared for each policy. The MYGA bridge
still uses a toolkit-only scalar Cell and the retained
formula/type specification to check performance. Neither route is an accepted
compiler implementation or proof of general exported-package compatibility.

The DV01/CS01 integration compiles its original `Actual`, `Valuation`, and
`Outer` Cell formulas as separate native artifacts. `load_connected` wires
their declared Space references in verified source-graph order;
`assemble_connected_package` copies them into one importable package with
source, Data, and native-binary hashes. The package shell loads and wires the
Spaces; it does not interpret Cell formulas. The collect domain selects the
policy parameter by name, even when another required Space parameter precedes
it. An importer supplies that parameter's internal initial value through
`parameter_initial_values`; the public argument stays required. Prepared
Valuation rows can use detached output from compiled `Actual`.

The seven-policy fixture and 105-policy compiled scaling check match all
three years and eight valuation scenarios per year, including the nested
`Actual.collect()` dependency and time-indexed `sum`/`individual` outputs.
The package still names the exported Python `ScenarioInputs` Space and the
rediscount helper as external providers. Building the three artifacts and
choosing their source-specific input preparation remain separate steps; the
standard `mx2cy --frame-collect` command still builds one Space. A paired
single-run timing found no established calculation speed change from package
assembly. Its 105-policy cost is a connected-calculation question, not a
package-import optimization result.

Windows portability edits in this snapshot are limited to UTF-8 generated
source reads/writes and a package-local generated facade import. The toolkit
campaign records the generated and loaded binary identities, full MYGA
cash-flow validation, and a focused context-ownership check.
