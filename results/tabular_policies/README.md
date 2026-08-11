# Final tabular policies

This directory records the selected optimised CFR, CFR+, and MCCFR average policies for Kuhn and Leduc poker. Training used fixed iteration budgets with early stopping disabled.

## Training budgets

Kuhn used 100,000 iterations. The 10,000-iteration correctness run had already reached approximately `0.000111` exploitability with ordinary CFR, so a 10x larger final run was sufficient to pass the stricter `0.0001` target while remaining inexpensive.

Leduc used 500,000 iterations. Ordinary CFR reached approximately `0.00334` exploitability after 5,000 iterations. CFR's general inverse-square-root convergence rate suggested that 100 times as many iterations should reduce this by roughly a factor of 10, placing it below the predeclared `0.0005` target. The same budget was used for CFR+ rather than tailoring the workload to its faster convergence.

CFR+ used an averaging delay of 10 iterations, matching its validated configuration. Both algorithms are deterministic here, so the recorded seed is compatibility metadata rather than a source of training variation.

Leduc MCCFR used 20,000,000 iterations across five fixed seeds. A 10,000,000-iteration calibration remained above the `0.005` validation ceiling; doubling the sampled work was consistent with MCCFR's usual inverse-square-root convergence trend and reached a median of `0.00450`. The median final seed was selected rather than the best seed. Full multi-seed evidence is in `results/mccfr/`.

## Exact evaluation

All values use the games' base chip unit and are reported per hand. Each player antes 1 chip, creating a 2-chip starting pot in both games. Kuhn bets are 1 chip; Leduc bets are 2 chips in the first round and 4 chips in the second round. Lower exploitability is better. Player 0 value is the expected number of chips won or lost per hand when both players use the saved policy.

| Game | Policy | Iterations | Player 0 value | Exploitability | NashConv | Target | Solver time |
|---|---|---:|---:|---:|---:|---:|---:|
| Kuhn | CFR | 100,000 | -0.05555525 | 0.00001719 | 0.00003437 | <= 0.0001 | 0.482 s |
| Kuhn | CFR+ | 100,000 | -0.05555556 | 0.00000215 | 0.00000431 | <= 0.0001 | 0.488 s |
| Leduc | CFR | 500,000 | -0.08561024 | 0.00014595 | 0.00029189 | <= 0.0005 | 80.391 s |
| Leduc | CFR+ | 500,000 | -0.08560642 | 0.00000017 | 0.00000034 | <= 0.0005 | 76.935 s |
| Leduc | MCCFR | 20,000,000 | -0.08639316 | 0.00450098 | 0.00900197 | <= 0.005 | 54.155 s |

Against Kuhn's known Player 0 equilibrium value of `-1/18` of the 1-chip ante per hand, the CFR value error was approximately `0.00000031` chips and the CFR+ value error was below `0.00000000005` chips.

The policies were evaluated with the independent exact best-response evaluator. Full-precision values and provenance identifiers are stored in `evaluations.csv` and `configs/strategy_registry.json`.

## Artifacts

The five selected snapshots are staged under ignored `artifacts/tabular/` paths and loaded through the strategy registry and `TabularAgent`. Only selected final snapshots are registered. File sizes and SHA-256 checksums are recorded there. Training checkpoints, milestone snapshots, metrics, and plots remain under ignored `runs/` directories.

The registry uses `local-tabular-final` as a temporary local artefact identifier. Publishing selected policy files and replacing this with a real download identifier are deferred until the web and packaging work.

The CFR/CFR+ runs used source revision `edc18f2e1dea52f3255f93d4e8612cb070bf2e83`. The MCCFR validation records its revision in `results/mccfr/validation.json`.
