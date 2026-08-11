# Reference Deep CFR validation on Leduc

**Status: PASS**

## Scope

This validation checks that the deliberately slow reference Deep CFR implementation learns in
the correct direction across multiple random seeds. It also exercises periodic average-strategy
snapshots, exact Leduc evaluation, checkpointing, compact metrics and held-out neural losses.

This is a correctness reference before optimisation. It is not the final Leduc Deep CFR policy
and does not claim near-equilibrium quality.

## Workload

Five seeds, `20260810` to `20260814`, each ran for 20 outer iterations. Every outer iteration
performed 200 external-sampling traversals for each player, or 400 traversals in total. Each seed
therefore completed 8,000 traversals.

The networks used three 64-unit hidden layers, Adam with a `0.001` learning rate, batches of 512,
10 advantage-training epochs and 20 average-strategy-training epochs. Reservoir capacities were
100,000 samples, 10% of each training set was held out for validation, gradient norms were capped
at 10, and dropout remained disabled. Early stopping was disabled.

The workload was selected after small calibration runs. It gives the reference implementation
roughly two minutes of median training work per seed and four separated evaluation points without
turning the intentionally inefficient implementation into a production-scale run. The later
optimised implementation will use larger workloads and determine final Leduc strategy quality.

Recorded training time includes the configured milestone average-strategy-network training
inside `solver.train()`. Exact evaluation, checkpoint writes, snapshot writes and plotting are
outside that time.

## Results

Every seed finished with lower exact exploitability than at its first snapshot. The median also
decreased at every declared milestone.

| Outer iterations | Traversals per seed | Median exploitability | Seed range | Median training time |
|---:|---:|---:|---:|---:|
| 1 | 400 | 2.7574 | 1.7466 to 3.9952 | 1.58 s |
| 5 | 2,000 | 1.9510 | 1.6981 to 1.9987 | 16.56 s |
| 10 | 4,000 | 1.0357 | 1.0067 to 1.1769 | 44.41 s |
| 20 | 8,000 | 0.8199 | 0.6679 to 0.9299 | 124.87 s |

The final median remains far above the validated tabular CFR/CFR+ ceiling of `0.005` chips per
hand. That gap is shown deliberately: this short reference run establishes consistent learning,
while optimisation and subsequent larger multi-seed validation are still required.

Training and held-out average-strategy losses remained similar at later milestones. This provides
no obvious conventional overfitting warning, but loss is only a diagnostic. Exact exploitability
is the strategy-quality measurement.

## Checkpoint and snapshot flow

Each run saved iteration-labelled checkpoints and a `latest.pt` alias. A checkpoint contains the
networks, reservoirs, deterministic RNG states, configuration, elapsed training time and compact
metric records needed for exact continuation.

At iterations 1, 5, 10 and 20, training froze a separate average-strategy network. Those snapshots
contain only inference requirements, are loaded with PyTorch's restricted weights-only loader,
and were converted into complete Leduc policies for independent exact evaluation. Advantage
networks and reservoirs are training state and are not exposed as playable policies.

## Files

- `convergence.csv`: full-precision measurements for every seed and snapshot milestone.
- `summary.csv`: medians and complete seed ranges at each milestone.
- `validation.json`: exact configuration, pass checks and links to run-local artefacts.
- `plots/reference_convergence.png`: per-seed and median exploitability by iterations and time.

The large checkpoints and neural snapshots remain under ignored `runs/` directories rather than
normal Git history. The compact validation evidence in this directory is version-controlled.
