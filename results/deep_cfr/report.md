# Deep CFR on Leduc

- **Reference learning validation:** PASS
- **Matched implementation convergence:** PASS
- **Implementation profiling and benchmark:** COMPLETE
- **Bounded configuration selection:** COMPLETE

## Scope

These results validate Deep CFR learning, compare the reference and optimised implementations, and select a bounded Leduc configuration. The selected setup has not yet undergone the moderate multi-seed validation stage.

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

Halving the advantage-training budget and shrinking the network materially reduced strategy quality. Raising the advantage budget to 150 or 200 steps improved this seed's exploitability by only 0.043 and 0.023 chips while increasing training time by 34% and 63%; 200 steps was also slightly worse than 150. The lower-K result was slightly better for this one seed, but one stochastic run does not show that half as many samples remains sufficient over longer training, and the saving was only 2.25 seconds. The selected Leduc configuration therefore retains `K = 1,000`, 100 advantage-training steps and the 64-unit network.

Baseline training and held-out losses remained close, so this run showed no clear conventional overfitting. Packed reservoirs used 50.2 MB; trained network parameters used about 131 KB, meaning the smaller network would not materially reduce total memory while reservoir storage dominates.

`leduc_selected.toml` freezes these learning settings for the next validation stage. Its 100-iteration budget and early/intermediate snapshots define the moderate validation run rather than another tuning variable.

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

Large checkpoints and neural snapshots remain under ignored `runs/` directories.
