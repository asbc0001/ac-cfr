# Deep CFR on Leduc

- **Reference learning validation:** PASS
- **Matched implementation convergence:** PASS
- **Implementation profiling and benchmark:** COMPLETE
- **Bounded configuration selection:** COMPLETE
- **Training lifecycle validation:** PASS

## Scope

These results validate Deep CFR learning, compare the reference and optimised implementations, and select a bounded Leduc configuration for final-policy training.

Training uses fixed minibatch-update budgets sampled uniformly from the reservoirs, so neural-training cost does not grow automatically with reservoir size.

## Learning and equivalence

The reference solver ran three seeds for 10 outer iterations. Median exact exploitability decreased at every milestone:

| Iteration | Traversals per seed | Median exploitability | Seed range |
|---:|---:|---:|---:|
| 1 | 400 | 3.4792 | 1.8533 to 3.8519 |
| 5 | 2,000 | 1.9776 | 1.9596 to 2.0180 |
| 10 | 4,000 | 1.7601 | 1.5746 to 2.0512 |

A separate fixed-seed comparison used identical architecture and training work for both implementations:

| Iteration | Traversals | Reference | Optimised |
|---:|---:|---:|---:|
| 1 | 2,500 | 2.9628 | 2.9695 |
| 2 | 5,000 | 2.1991 | 2.1829 |
| 4 | 10,000 | 1.7559 | 1.7858 |

Both implementations improve along closely matched trajectories. Their final exploitabilities differ by `0.0299` chips, within the declared `0.1`-chip behavioural tolerance. The comparison uses 140 optimizer steps because it trains a strategy network at all three milestones; the performance benchmark below trains only the final strategy network.

These short validation policies remain far above the validated tabular ceiling of `0.005` chips per hand. They establish correct learning behaviour, not final Deep CFR capability.

## Initial configuration study

Six optimised runs changed one factor at a time over 20 outer iterations. This is a bounded sensitivity check, not an exhaustive search or a claim that the selected values are optimal.

| Configuration | Changed factor | Training time | Exact exploitability |
|---|---|---:|---:|
| Baseline | None | 12.89 s | 1.0711 |
| Lower K | 1,000 to 500 traversals per player | 10.63 s | 1.0142 |
| Fewer advantage updates | 100 to 50 steps | 11.05 s | 1.8438 |
| 150 advantage updates | 100 to 150 steps | 17.23 s | 1.0284 |
| 200 advantage updates | 100 to 200 steps | 21.01 s | 1.0484 |
| Smaller network | 64 to 32 units per hidden layer | 12.72 s | 1.6884 |

Halving the advantage-training budget and shrinking the network materially reduced strategy quality. At 20 iterations, raising the advantage budget to 150 or 200 steps gave only a small improvement for extra cost, so the initial selected setup retained 100 steps. The lower-K result was slightly better for this one seed, but one stochastic run does not show that half as many samples remains sufficient over longer training.

Baseline training and held-out losses remained close, so this run showed no clear conventional overfitting. Packed reservoirs used 50.2 MB; trained network parameters used about 131 KB, meaning the smaller network would not materially reduce total memory while reservoir storage dominates.

The longer validation then exposed a limitation the 20-iteration study could not: with 100 advantage updates, exploitability improved to 1.0711 at iteration 20 but worsened to 1.3728 by iteration 100 while advantage training and held-out losses rose together. A bounded 50-iteration diagnostic showed that 200 updates sustained learning, so `leduc_selected.toml` was corrected to 200 advantage updates without changing `K`, model size, learning rate or other settings.

## Training lifecycle validation

Three optimised seeds used the interim 200-update settings through iteration 20. Every seed improved, and median exact exploitability fell from 3.2883 to 1.0484 chips.

| Seed | Iteration 1 | Iteration 20 |
|---:|---:|---:|
| 20260810 | 4.0441 | 1.0697 |
| 20260811 | 3.2883 | 1.0484 |
| 20260812 | 1.6958 | 0.9351 |

The moderate seed was deliberately interrupted after its iteration-20 checkpoint and successfully resumed. Its later exact results were:

| Iteration | Training time | Exact exploitability |
|---:|---:|---:|
| 1 | 1.16 s | 3.2883 |
| 20 | 19.29 s | 1.0484 |
| 50 | 50.15 s | 1.0507 |
| 100 | 98.04 s | 0.9013 |

At this stage, iteration-100 exploitability was lower than at iteration 20 and final advantage losses had also fallen. The moderate run resumed successfully from its intentional iteration-20 interruption, and all eight exported snapshots matched `NeuralAgent` probabilities across every Leduc information set within `1e-6`. This validated the complete training lifecycle, but later diagnostics showed that the neural-training budgets were still too small for the final policy.

## Final-configuration precheck

Longer diagnostics showed that the earlier 200-step advantage and strategy budgets undertrained the neural approximations. The final preset therefore uses 1,000 advantage updates per player and 1,000 updates per exported strategy network, with gradient clipping reduced from 10.0 to 1.0. It retains 1,000 traversals per player, the 64-unit network and the existing reservoir, batch-size and learning-rate settings.

Two independent 20-iteration checks confirmed the revised configuration learns consistently before the final run:

| Seed | Iteration 1 exploitability | Iteration 20 exploitability | Training time |
|---:|---:|---:|---:|
| 20260810 | 4.0035 | 0.3775 | 74.60 s |
| 20260812 | 1.6933 | 0.4401 | 77.27 s |

All recorded losses were finite, and final training and held-out losses remained comparable. A separate 150-iteration diagnostic reached 0.2295 exact exploitability; its weighted strategy reservoir reached 0.1910. Doubling the hidden-layer width did not improve the exported strategy, so the final preset retains the leaner 64-unit architecture. These checks select a reasonable configuration; they are not the final policy run.

## Engineering results

The fixed benchmark gives each implementation 10,000 traversals and 100 optimizer steps. Three warmed fresh-process repetitions time only training; evaluation, plotting and startup are excluded. Both use one PyTorch CPU thread because these small Leduc networks run faster without multi-thread scheduling overhead.

| Measurement | Reference | Optimised |
|---|---:|---:|
| Median training time | 9.72 s | 1.48 s |
| Timing MAD | 0.95 s | 0.13 s |
| Traversal collection rate | 1,098/s | 7,998/s |
| Peak process-tree PSS | 773.1 MB | 654.1 MB |

This is a `6.55x` training speedup, `7.29x` higher traversal throughput and `15.4%` lower peak memory for the declared CPU workload. MAD measures typical timing variation around the median. PSS measures process memory without fully double-counting shared pages.

Profiling supports the timing result: batching reduces network calls from 30,315 to 365, packed handling reduces Python-visible calls from 19.3 million to 1.5 million, and optimised traversal uses about 0.6 profiled seconds versus 6.3 seconds for recursive reference traversal. Profiled times themselves are diagnostic because instrumentation changes execution speed.

## Reproducibility

Reference runs exported independently loadable policies at iterations 1, 5 and 10. Their checkpoints contain networks, reservoirs, random-number-generator states, resolved configuration, elapsed training time and metrics required for compatible resume.

- `validation.json`, `convergence.csv`, and `summary.csv` contain the multi-seed reference evidence.
- `comparison.json`, `implementation_convergence.csv`, and `plots/implementation_convergence.png` contain the matched implementation comparison.
- `benchmark.json`, `benchmark_*.csv`, and `plots/implementation_performance.png` contain the performance evidence.
- `profiling.json` and `profiles/` contain the diagnostic workloads and profiles.
- `configuration_study.json`, `configuration_study.csv`, and `plots/configuration_sensitivity.png` contain the bounded configuration study.
- `selected_validation.json`, `selected_convergence.csv`, and `plots/selected_validation.png` contain the selected-configuration lifecycle check.
- `final_configuration_checks.csv` contains the two short checks of the frozen final preset.

Large checkpoints and neural snapshots remain under ignored `runs/` directories.
