# Importance-corrected exploratory sampling on Leduc

This experiment samples Leduc opponent actions from `q = 0.9 sigma + 0.1 uniform` and corrects the distribution shift with importance weights.

- Estimator validation: **PASS**
- Matched three-seed training gate: **FAIL**
- Modified-HULHE runs: **NOT RUN**

Only opponent sampling changes. Chance sampling and traverser expansion are unchanged; local `sigma/q` ratios correct returns, cumulative prefix ratios weight samples, strategy memory stores `sigma`, and reservoir replacement remains uniform.

The 100,000-sample estimator check against the complete Leduc tree passed:

| Measurement | Result | Gate |
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

Median exploitability improved slightly, but only one paired seed improved, so the declared Leduc gate failed. The estimator remains validated; a modified-HULHE-specific hypothesis would require a separate predeclared test.

Evidence: [`estimator_validation.json`](estimator_validation.json), [`training_results.csv`](training_results.csv), [`training_summary.csv`](training_summary.csv), and [`validation.json`](validation.json).
