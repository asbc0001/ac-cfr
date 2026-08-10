# Final tabular policies

These are the selected optimised CFR and CFR+ average policies for Kuhn and Leduc poker. Training used fixed iteration budgets with early stopping disabled, so the runs are reproducible and directly comparable within each game.

## Training budgets

Kuhn used 100,000 iterations. The 10,000-iteration correctness run had already reached approximately `0.000111` exploitability with ordinary CFR, so a 10x larger final run was sufficient to pass the stricter `0.0001` target while remaining inexpensive.

Leduc used 500,000 iterations. Ordinary CFR reached approximately `0.00334` exploitability after 5,000 iterations. CFR's general inverse-square-root convergence rate suggested that 100 times as many iterations should reduce this by roughly a factor of 10, placing it below the predeclared `0.0005` target. The same budget was used for CFR+ rather than tailoring the workload to its faster convergence.

CFR+ used an averaging delay of 10 iterations, matching its validated configuration. Both algorithms are deterministic here, so the recorded seed is compatibility metadata rather than a source of training variation.

## Exact evaluation

Lower exploitability is better. Player 0 value is the expected number of chips won or lost per hand when both players use the saved policy.

| Game | Policy | Iterations | Player 0 value | Exploitability | NashConv | Target | Solver time |
|---|---|---:|---:|---:|---:|---:|---:|
| Kuhn | CFR | 100,000 | -0.05555525 | 0.00001719 | 0.00003437 | <= 0.0001 | 0.482 s |
| Kuhn | CFR+ | 100,000 | -0.05555556 | 0.00000215 | 0.00000431 | <= 0.0001 | 0.488 s |
| Leduc | CFR | 500,000 | -0.08561024 | 0.00014595 | 0.00029189 | <= 0.0005 | 80.391 s |
| Leduc | CFR+ | 500,000 | -0.08560642 | 0.00000017 | 0.00000034 | <= 0.0005 | 76.935 s |

Against Kuhn's known Player 0 equilibrium value of `-1/18`, the CFR value error was approximately `0.00000031` chips and the CFR+ value error was below `0.00000000005` chips.

The policies were evaluated with the independent exact best-response evaluator. Full-precision values and provenance identifiers are stored in `evaluations.csv` and `configs/strategy_registry.json`.

## Artifacts

The four selected snapshots are staged under ignored `artifacts/tabular/` paths and loaded through the strategy registry and `TabularAgent`. Kuhn snapshots were exported at 10,000, 50,000, and 100,000 iterations; Leduc snapshots were exported at 50,000, 200,000, and 500,000 iterations. Only the final snapshot from each run was selected for the registry. File sizes and SHA-256 checksums are recorded there. Training checkpoints, milestone snapshots, metrics, and plots remain under ignored `runs/` directories.

The registry uses `local-tabular-final` as a temporary local artefact identifier. Publishing selected policy files and replacing this with a real download identifier are deferred until the web and packaging work.

All runs used source revision `edc18f2e1dea52f3255f93d4e8612cb070bf2e83`.
