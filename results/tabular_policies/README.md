# Final tabular policies

This directory contains exact evaluation and provenance for the selected optimised CFR, CFR+ and MCCFR average policies on Kuhn and Leduc. Training used fixed budgets with early stopping disabled.

## Training budgets

| Game | Policy | Iterations | Selection basis |
|---|---|---:|---|
| Kuhn | CFR and CFR+ | 100,000 | Ten times the validated 10,000-iteration workload remained inexpensive and passed the stricter 0.0001 target |
| Leduc | CFR and CFR+ | 500,000 | CFR's measured convergence supported increasing the validated workload by 100 times to pass the 0.0005 target |
| Leduc | MCCFR | 20,000,000 across five seeds | Ten million iterations remained above 0.005; doubling the sampled work produced a 0.00450 median |

CFR+ used an averaging delay of 10. CFR and CFR+ are deterministic here, while MCCFR quality was evaluated across five seeds and the median final seed was selected rather than the best. Full MCCFR evidence is in the [Leduc MCCFR results](../leduc_mccfr/).

## Exact evaluation

Values are chips per hand. Each player antes one chip; Kuhn bets are one chip, while Leduc bets are two chips in the first round and four in the second. Lower exploitability is better.

NashConv sums both players' possible improvement from switching individually to a best response; exploitability is half of NashConv here.

| Game | Policy | Iterations | Player 0 value | Exploitability | NashConv | Target | Solver time |
|---|---|---:|---:|---:|---:|---:|---:|
| Kuhn | CFR | 100,000 | -0.05555525 | 0.00001719 | 0.00003437 | ≤ 0.0001 | 0.482 s |
| Kuhn | CFR+ | 100,000 | -0.05555556 | 0.00000215 | 0.00000431 | ≤ 0.0001 | 0.488 s |
| Leduc | CFR | 500,000 | -0.08561024 | 0.00014595 | 0.00029189 | ≤ 0.0005 | 80.391 s |
| Leduc | CFR+ | 500,000 | -0.08560642 | 0.00000017 | 0.00000034 | ≤ 0.0005 | 76.935 s |
| Leduc | MCCFR | 20,000,000 | -0.08639316 | 0.00450098 | 0.00900197 | ≤ 0.005 | 54.155 s |

Against Kuhn's known Player 0 equilibrium value of `-1/18`, CFR's error was approximately 0.00000031 chips and CFR+'s was below 0.00000000005 chips. The independent exact best-response evaluator produced the full-precision [`evaluations.csv`](evaluations.csv) records.

## Artifacts

The five selected snapshots are staged under ignored `artifacts/tabular/` paths and loaded through the strategy registry and `TabularAgent`. File sizes, SHA-256 checksums and compatibility metadata are recorded in [`configs/strategy_registry.json`](../../configs/strategy_registry.json).

The registry still uses the temporary `local-tabular-final` release identifier. Publishing the policy files and replacing that identifier are deferred until release preparation. CFR/CFR+ used source revision `edc18f2e1dea52f3255f93d4e8612cb070bf2e83`; MCCFR provenance is in [`validation.json`](../leduc_mccfr/validation.json).
