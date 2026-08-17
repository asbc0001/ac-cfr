# Leduc Deep CFR

This directory contains correctness, configuration, convergence, performance and final-policy evidence for the reference and optimised Leduc Deep CFR implementations.

- **Reference learning and matched implementation validation:** PASS
- **Interruption/resume and playable-policy validation:** PASS
- **Final Leduc policy:** COMPLETE
- **Exploratory-sampling estimators:** PASS
- **Exploratory-sampling training comparison:** FAIL

## Learning and implementation validation

Three reference seeds improved over ten outer iterations:

| Iteration | Traversals per seed | Median exploitability | Seed range |
|---:|---:|---:|---:|
| 1 | 400 | 3.4792 | 1.8533 to 3.8519 |
| 5 | 2,000 | 1.9776 | 1.9596 to 2.0180 |
| 10 | 4,000 | 1.7601 | 1.5746 to 2.0512 |

A fixed-seed comparison then gave the reference and optimised implementations identical architecture and training work:

| Iteration | Traversals | Reference exploitability | Optimised exploitability |
|---:|---:|---:|---:|
| 1 | 2,500 | 2.9628 | 2.9695 |
| 2 | 5,000 | 2.1991 | 2.1829 |
| 4 | 10,000 | 1.7559 | 1.7858 |

Their final difference was 0.0299 chips, inside the declared 0.1-chip behavioural tolerance. These short runs validate learning and implementation agreement; they are not competitive final policies.

## Configuration and training recovery

An initial six-run sensitivity study changed one factor at a time for 20 iterations. Halving the advantage-update budget or network width materially worsened exploitability. Raising advantage updates from 100 to 150 or 200 gave only small early improvements, while the 64-unit network and 1,000 traversals per player remained appropriate.

Longer training exposed underfitting that the 20-iteration study did not: the 100-update configuration improved to 1.0711 exploitability at iteration 20 but worsened to 1.3728 by iteration 100. A follow-up run with 200 updates restored progress. A separate interruption test resumed successfully from iteration 20, preserved continuous metrics and exported policies that matched `NeuralAgent` across every Leduc information set within `1e-6`.

Further tests found that 200 updates still undertrained the neural approximations. The final configuration therefore used 1,000 advantage updates per player and 1,000 strategy-network updates, retained 1,000 traversals per player and the three-layer 64-unit network, and reduced gradient clipping from 10.0 to 1.0. Two independent runs confirmed consistent early learning:

| Seed | Iteration 1 exploitability | Iteration 20 exploitability | Training time |
|---:|---:|---:|---:|
| 20260810 | 4.0035 | 0.3775 | 74.60 s |
| 20260812 | 1.6933 | 0.4401 | 77.27 s |

All recorded losses were finite, and training and held-out losses remained comparable. A separate 150-iteration run reached 0.2295 exploitability. Doubling the hidden-layer width did not improve the exported policy.

## Final policy

The final optimised run completed 200 training iterations and 400,000 sampled traversals in 770.16 seconds. Exact evaluation was performed at the saved iterations below:

| Iteration | Training time | Exact exploitability |
|---:|---:|---:|
| 1 | 6.56 s | 3.6753 |
| 20 | 82.05 s | 0.4003 |
| 50 | 192.69 s | 0.2559 |
| 75 | 285.66 s | 0.2131 |
| 100 | 387.22 s | 0.3752 |
| 125 | 486.21 s | 0.3108 |
| 150 | 579.82 s | **0.2056** |
| 175 | 671.45 s | 0.2801 |
| 200 | 770.16 s | 0.2503 |

Iteration 150 had the lowest measured exploitability and was selected instead of the final snapshot. Its exact Player 0 value is -0.130282 chips per hand and its NashConv is 0.411299 chips. NashConv sums both players' possible improvement from switching individually to a best response. `NeuralAgent` reproduced its probabilities across all 936 Leduc information sets within `1e-6`.

The selected snapshot is staged at `artifacts/deep_cfr/leduc-deep-cfr-final.pt` and registered as `leduc_deep_cfr_final`. Iterations 20 and 75 provide the registered early and intermediate policies. The run used source revision `af079e3cc360be130b7ee5824ddcef8b5ddf9e4a`.

## Engineering results

The fixed CPU benchmark gave each implementation 10,000 traversals and 100 optimiser steps. Three warmed fresh-process repetitions timed training only, using one PyTorch CPU thread.

| Measurement | Reference | Optimised |
|---|---:|---:|
| Median training time | 9.72 s | 1.48 s |
| Typical timing variation (MAD) | 0.95 s | 0.13 s |
| Traversal collection rate | 1,098/s | 7,998/s |
| Peak proportional memory (PSS) | 773.1 MB | 654.1 MB |

MAD is the median absolute deviation from the median runtime. PSS divides shared memory among the processes using it. The optimised implementation was 6.55 times faster, collected traversals 7.29 times faster and used 15.4% less peak memory for this workload. Profiling reduced network calls from 30,315 to 365 and Python-visible calls from 19.3 million to 1.5 million. The complete final run averaged 519.4 traversals per second after neural training and snapshot work were included.

## Exploratory opponent sampling

The separate Leduc experiment mixed the current policy (`sigma`) with 10% uniform exploration, producing the sampling policy `q = 0.9 sigma + 0.1 uniform`. Importance weights corrected the resulting samples. Chance sampling, the updating player's action expansion, true-policy storage and uniform replacement in the bounded training memories were unchanged. Zero exploration retained the original behaviour.

The 100,000-sample estimator check passed: advantage reach-weighted RMSE was 0.1094, strategy-memory RMSE was 0.00244, all 1,872 expected information sets were observed, and effective sample size was 99.78% of the weighted sample count.

The matched 20-iteration comparison did not meet its improvement requirement:

| Seed | Baseline exploitability | Exploratory exploitability | Improved |
|---:|---:|---:|---|
| 20260811 | 1.0711 | 1.0914 | No |
| 20260812 | 1.1611 | 1.2812 | No |
| 20260813 | 1.1077 | 1.0725 | Yes |

Only one paired seed improved, below the required two seeds. Later modified-HULHE experiments also failed to justify 10% exploration for the final run, which used no added exploration. Detailed Leduc evidence is in [`exploratory_sampling/`](exploratory_sampling/); the modified-HULHE outcome is summarised separately in [`../modified_hulhe/exploratory_sampling/`](../modified_hulhe/exploratory_sampling/).

## Evidence files

The generated files remain flat because the benchmark suites share this output directory.

- [`validation.json`](validation.json), [`convergence.csv`](convergence.csv) and [`summary.csv`](summary.csv): multi-seed reference validation.
- [`comparison.json`](comparison.json), [`implementation_convergence.csv`](implementation_convergence.csv) and [`plots/implementation_convergence.png`](plots/implementation_convergence.png): matched implementation comparison.
- [`configuration_study.json`](configuration_study.json), [`selected_validation.json`](selected_validation.json), [`final_configuration_checks.csv`](final_configuration_checks.csv) and the corresponding plots: configuration and interruption/resume evidence.
- [`final_policy.json`](final_policy.json), [`final_policy_convergence.csv`](final_policy_convergence.csv), [`plots/final_policy_convergence.png`](plots/final_policy_convergence.png) and [`lifecycle_gate.json`](lifecycle_gate.json): final-run provenance, evaluated iterations and completeness checks.
- [`benchmark.json`](benchmark.json), [`profiling.json`](profiling.json), [`profiles/`](profiles/), [`plots/implementation_performance.png`](plots/implementation_performance.png) and [`traversal_scaling.json`](traversal_scaling.json): performance and profiling evidence.
- [`exploratory_sampling/`](exploratory_sampling/): estimator and matched training evidence for the experimental sampler.

Large checkpoints and neural snapshots belong under ignored `runs/` directories rather than this compact evidence set.
