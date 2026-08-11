# Deep CFR on Leduc

- **Reference learning validation:** PASS
- **Implementation profiling:** COMPLETE

## Scope

This evidence checks that the reference Deep CFR implementation learns in the correct direction
across several seeds and identifies the main costs in the reference and optimised implementations.
It is not the final Leduc policy or the formal performance comparison.

Deep CFR now uses fixed optimizer-step budgets. Each training step draws one minibatch uniformly,
with replacement, from the training portion of the relevant uniform reservoir. This follows the
intended reservoir distribution while preventing later outer iterations from becoming more
expensive merely because more samples have been collected.

## Reference learning validation

Three seeds, `20260810` to `20260812`, each ran for 10 outer iterations. Every iteration performed
200 external-sampling traversals per player, giving 4,000 traversals per seed. Each freshly
initialised advantage network received 20 minibatch updates. Average-strategy networks exported
at iterations 1, 5 and 10 received 40 updates each, giving 520 optimizer steps per seed.

The networks used three 64-unit hidden layers, Adam with a `0.001` learning rate and batches of
512. Reservoir capacities were 100,000 samples, 10% of each current reservoir was held out for
validation, gradient norms were capped at 10, and dropout remained disabled.

All three seeds ended with lower exact exploitability than at their first snapshot, and the median
decreased at every predeclared milestone.

| Outer iterations | Traversals per seed | Median exploitability | Seed range | Median training time |
|---:|---:|---:|---:|---:|
| 1 | 400 | 3.4792 | 1.8533 to 3.8519 | 10.81 s |
| 5 | 2,000 | 1.9776 | 1.9596 to 2.0180 | 37.00 s |
| 10 | 4,000 | 1.7601 | 1.5746 to 2.0512 | 65.91 s |

The final median remains well above the validated tabular exploitability ceiling of `0.005` chips
per hand. That is expected for this bounded reference check. A later moderate optimised-only run
will assess practical Leduc strategy quality.

Training and held-out losses use separate samples and remain broadly comparable. They can reveal
ordinary overfitting, but exact exploitability is the strategy-quality measurement.

## Checkpoints and snapshots

Each run saved iteration-labelled checkpoints and a `latest.pt` alias. A checkpoint contains the
networks, reservoirs, deterministic random-number-generator states, configuration, elapsed
training time and compact metrics needed to continue from a completed outer iteration.

At iterations 1, 5 and 10, training froze a separate average-strategy network. Each snapshot was
loaded independently and converted into a complete Leduc policy for exact evaluation. Advantage
networks and reservoirs remain training state and are not playable policies.

## Implementation profiling

Both solvers were warmed before profiling. The `cProfile` workload used the same seed, network,
batch size, three outer iterations, 3,000 traversals and 220 optimizer steps for each
implementation. It took 31.72 profiled seconds for the reference solver and 30.13 for the
optimised solver. These instrumented times identify bottlenecks and are not formal speed results.

The separate PyTorch operator workload used 500 traversals and 60 optimizer steps. It was large
enough to include repeated inference, forward passes, backpropagation and Adam updates while
remaining short. Tensor-shape and memory tracing were disabled because their extra overhead was
not needed for this question.

The profiles show:

- fixed neural training dominates both implementations, with backpropagation, matrix operations
  and Adam updates accounting for most CPU time;
- batched inference reduces network forward calls from 30,315 to 365;
- packed storage and batching reduce Python-visible calls from 19.3 million to 1.5 million; and
- optimised traversal occupies about 0.6 cumulative profiled seconds, compared with about 6.3
  seconds for recursive reference traversal.

The optimised traversal path is materially leaner, but the identical neural-training workload now
dominates end-to-end time. Multiprocessing is therefore not justified for this small Leduc CPU
workload. Formal repeated timing, throughput and memory measurement remains separate from these
diagnostic profiles.

## Files

- `convergence.csv`: full-precision exact measurements for every seed and milestone.
- `summary.csv`: medians and complete seed ranges at each milestone.
- `validation.json`: configuration, checks and links to run-local checkpoints and snapshots.
- `plots/reference_convergence.png`: per-seed and median exploitability by iterations and time.
- `profiling.json`: exact profile workloads, environment and profile file index.
- `profiles/*_cprofile.md`: Python and native-call CPU paths ranked by cumulative time.
- `profiles/*_torch_profiler.md`: PyTorch operations ranked by self CPU time.

Large checkpoints and neural snapshots remain under ignored `runs/` directories. The compact
validation and profiling evidence in this directory is suitable for version control.
