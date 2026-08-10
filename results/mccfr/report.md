# MCCFR validation and implementation benchmark on Leduc

**Status: PASS**

## Scope

This work compares the reference and optimised external-sampling MCCFR implementations, checks the optimised solver with exact Leduc strategy evaluation, verifies the exported playable policy, and measures implementation performance under a fixed workload.

Results are in chips per hand. Each player antes 1 chip, creating a 2-chip starting pot; bets are 2 chips in the first round and 4 chips in the second. Lower exploitability is better.

Measured environment: 8 available CPUs, 15.4 GiB RAM, Python 3.12.3 on `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.

## Workload and rationale

Both implementations use five fixed seeds, `20260810` to `20260814`. At the shared 500,000-iteration workload, the committed reference results are reused rather than needlessly rerun. The optimised solver continues to 20,000,000 iterations per seed.

A 10,000,000-iteration calibration produced about `0.00692` exploitability, above the established `0.005` CFR/CFR+ validation ceiling. MCCFR's usual inverse-square-root sampling trend suggested that doubling the work would reach roughly `0.0049`, making 20,000,000 a proportionate final workload rather than an arbitrary large run. Early stopping was disabled.

One MCCFR iteration performs two sampled traversals, one for each player. MCCFR and full-tree CFR iterations represent very different work and must not be compared directly.

The performance benchmark uses 500,000 iterations, or 1,000,000 sampled traversals, with seed 0 for both implementations. This is a shared validation milestone, gives the optimised solver a timed region above one second, and keeps five reference repeats practical. Repeated timings use the same seed so they measure system variation, while the separate five-seed runs above measure stochastic strategy variation.

## Results

When both implementations received identical random draws for 10 iterations, their regrets, strategy sums, current policies, and average policies agreed within `1e-12`. Ten iterations are enough for this semantic check because it tests each update step, not convergence.

At the shared 500,000 iterations, reference MCCFR had median exploitability `0.02985` across seeds. Optimised MCCFR produced `0.03080`, inside the reference seed range of `0.02676` to `0.03651`. This shows equivalent stochastic strategy quality for equal sampled work.

| Optimised iterations | Median exploitability | Seed range | Median Player 0 value |
|---:|---:|---:|---:|
| 500,000 | 0.03080 | 0.02725 to 0.03542 | -0.08794 |
| 2,000,000 | 0.01504 | 0.01478 to 0.01623 | -0.08756 |
| 5,000,000 | 0.00920 | 0.00893 to 0.01075 | -0.08682 |
| 10,000,000 | 0.00692 | 0.00646 to 0.00747 | -0.08658 |
| 20,000,000 | 0.00450 | 0.00434 to 0.00508 | -0.08627 |

The final median passed the `0.005` validation ceiling. For context, the selected final CFR and CFR+ policies have exploitabilities of approximately `0.000146` and `0.00000017`; MCCFR is noisier because it learns from sampled branches rather than the complete tree on every iteration.

The seed with the median final exploitability, `20260810`, was selected instead of the best seed. Its saved policy has exact exploitability `0.004501`, NashConv `0.009002`, and Player 0 value `-0.086393` chips per hand.

The exported policy was loaded from its saved snapshot and played against itself over 20,000 duplicate pairs, or 40,000 hands. Its mean was `-0.008525` chips per hand with a 99% confidence interval from `-0.035714` to `0.018664`, which includes neutral play at zero.

## Fixed-workload performance

| Iterations | Reference median | Reference relative MAD | Optimised median | Optimised relative MAD | Speedup |
|---:|---:|---:|---:|---:|---:|
| 500,000 | 53.2397 s | 0.82% | 1.2479 s | 1.02% | 42.7x |

| Reference traversals/s | Optimised traversals/s | Reference peak PSS | Optimised peak PSS | Memory change |
|---:|---:|---:|---:|---:|
| 18,783 | 801,326 | 73.27 MB | 115.58 MB | +57.7% |

The optimised solver is `42.7x` faster on this workload. MAD means median absolute deviation: the typical distance of a repeat from the median. Five fresh-process repeats were used. Only `solver.train()` was timed; process startup, Numba compilation, exact evaluation, plotting, and file writes were excluded. Numba was warmed up before every optimised timing.

Peak memory is process-tree PSS sampled every 10 ms. PSS assigns each process its private memory plus a proportional share of memory shared with other processes. The optimised solver used about 58% more memory because the warmed Numba runtime and compiled code remained resident.

The benchmark does not use its single fixed seed to judge strategy quality. That would confuse timing repeatability with sampling variation. Strategy quality instead comes from the five-seed exact evaluation described above.

## Profiler evidence

Separate `cProfile` runs use 100,000 iterations, or 200,000 sampled traversals. This is long enough to expose stable hot paths while keeping profiler overhead outside the formal timings.

The reference profile records about 40.9 million Python calls. Most time is in recursive tree traversal, sampling, policy normalisation, enum conversion, and Python collection operations. The optimised profile shows only 85 Python-level calls because traversal and updates execute inside one Numba-compiled machine-code boundary. `cProfile` cannot list individual operations inside that compiled function.

## Completion gate

The gate passed. Deterministic seeded tests cover sampling, regret updates, utility perspective, average-strategy accumulation, alternating player updates, and identical-draw agreement between implementations. The five-seed validation establishes comparable stochastic behaviour and convergence. Training workflow tests cover checkpoint resume, snapshot export, checksum validation, registry resolution, and playable-agent loading. Duplicate-deal self-play checks the exported policy's seat symmetry. The fixed benchmark and both representative profiles completed without changing the MCCFR update or sampling rules.

## Files

- `convergence.csv`: every exact measurement by implementation, seed, and milestone.
- `summary.csv`: medians and complete seed ranges at each milestone.
- `validation.json`: multi-seed workload, checks, selected snapshot metadata, and full-precision results.
- `benchmark_runs.csv`: every timing and peak-memory repetition.
- `benchmark_summary.csv`: median performance, variation, memory, and final fixed-seed quality.
- `plots/`: convergence and implementation-performance figures.
- `profiles/`: automatically generated reference and optimised CPU profiles.
- `gate.json`: configuration, environment, evidence index, and overall pass status.

The playable policy is stored locally at `artifacts/tabular/leduc-mccfr-final.npz` and registered as `leduc_mccfr_final`. The matching resumable checkpoint is under the ignored `runs/` directory.
