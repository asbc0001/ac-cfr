# Deep CFR on Leduc

- **Reference learning validation:** PASS
- **Matched implementation convergence:** PASS
- **Implementation profiling and benchmark:** COMPLETE
- **Bounded configuration selection:** COMPLETE
- **Selected-configuration validation:** PASS

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

## Configuration selection

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

## Selected-configuration validation

Three optimised seeds used the corrected selected settings through iteration 20. Every seed improved, and median exact exploitability fell from 3.2883 to 1.0484 chips.

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

The corrected setup passed the sustained-learning gate: iteration-100 exploitability was lower than at iteration 20, and final advantage losses also fell. The moderate run resumed successfully from its intentional iteration-20 interruption. All eight exported snapshots reloaded and matched `NeuralAgent` probabilities across every Leduc information set within `1e-6`.

This validates the selected setup before the separate, longer final-policy run; these validation snapshots are not presented as the final web policy.

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

Large checkpoints and neural snapshots remain under ignored `runs/` directories.
