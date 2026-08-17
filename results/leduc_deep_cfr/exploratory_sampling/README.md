# Importance-corrected exploratory sampling on Leduc

This experiment mixes the current Leduc policy (`sigma`) with 10% uniform exploration, producing the sampling policy `q = 0.9 sigma + 0.1 uniform`. Importance weights correct the resulting distribution shift.

- Estimator validation: **PASS**
- Matched three-seed training comparison: **FAIL**
- Exploratory modified-HULHE runs: **COMPLETE; NOT SELECTED**

Only opponent sampling changes. Chance sampling and the updating player's action expansion are unchanged; `sigma/q` ratios correct returns and sample weights, strategy memory stores the true policy, and replacement in the bounded training memories remains uniform.

The 100,000-sample estimator check against the complete Leduc tree passed:

| Measurement | Result | Acceptance threshold |
|---|---:|---:|
| Advantage reach-weighted RMSE | 0.109384 | <= 0.15 |
| Strategy-memory reach-weighted RMSE | 0.002441 | <= 0.005 |
| Information sets observed | 1,872 / 1,872 | complete |
| Maximum local importance ratio | 1.034483 | <= 1.111112 |
| Effective sample fraction | 99.78% | >= 80% |

The matched experiment used three seeds, 20 iterations and 1,000 traversals per player per iteration:

| Seed | Baseline final exploitability | Exploratory final exploitability |
|---:|---:|---:|
| 20260811 | 1.071071 | 1.091391 |
| 20260812 | 1.161080 | 1.281205 |
| 20260813 | 1.107745 | 1.072531 |

Median exploitability improved slightly, but only one paired seed improved instead of the required two. The estimator remained validated. The later [modified-HULHE experiment](../../modified_hulhe/exploratory_sampling/) did not justify exploratory sampling for the final run.

Evidence: [`estimator_validation.json`](estimator_validation.json), [`training_results.csv`](training_results.csv), [`training_summary.csv`](training_summary.csv), and [`validation.json`](validation.json).
