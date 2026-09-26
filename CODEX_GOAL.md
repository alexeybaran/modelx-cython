# Recommended Codex Goal

Use one durable project Goal for the end state; use Plan mode for individual experiments.

Suggested command:

```text
/goal Evolve pristine modelx-cython toward the previous optimized compiler's calculation performance on lifelib workloads where that compiler achieved native execution. Verify progress with untouched upstream CI, original and adjusted lifelib correctness, matched same-machine benchmarks, generated-code evidence where relevant, and later cross-model checks. Preserve the modelx-cython Space/Cell API and Cython-based architecture, keep compiler logic generic, and promote only changes with demonstrated material performance benefit or necessary generic correctness/coverage functionality. Experiments must stay off core; if a change qualifies for promotion, stop and request user approval before integrating it. If a material residual remains, attribute it before trying another mechanism.
```

Do not mark this Goal complete merely because MYGA reaches a historical number. Completion requires an accepted `core` satisfying the project definition of success, including cross-model evidence on lifelib workloads previously native in the previous compiler.
