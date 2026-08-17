# Leduc MCCFR

This directory contains correctness, convergence, final-policy, profiling and fixed-workload performance evidence for the reference and optimised external-sampling MCCFR implementations.

**Status: PASS**

## Validation and final policy

Exploitability and Player 0 values are chips per hand; lower exploitability is better. Each player antes one chip, and Leduc bets are two chips in the first round and four in the second.

When both implementations received identical random draws for ten iterations, their regrets, strategy sums, current policies and average policies agreed within `1e-12`. At the shared 500,000-iteration workload, reference median exploitability was 0.02985 across five seeds; the optimised result was 0.03080, inside the reference range of 0.02676 to 0.03651.

The optimised solver then continued to 20 million iterations for seeds 20260810–20260814:

| Iterations | Median exploitability | Seed range | Median Player 0 value |
|---:|---:|---:|---:|
| 500,000 | 0.03080 | 0.02725 to 0.03542 | -0.08794 |
| 2,000,000 | 0.01504 | 0.01478 to 0.01623 | -0.08756 |
| 5,000,000 | 0.00920 | 0.00893 to 0.01075 | -0.08682 |
| 10,000,000 | 0.00692 | 0.00646 to 0.00747 | -0.08658 |
| 20,000,000 | **0.00450** | 0.00434 to 0.00508 | -0.08627 |

The final median passed the 0.005 validation ceiling. Seed 20260810 was selected because it produced the median result rather than the best result. Its saved policy has exact exploitability 0.004501, NashConv 0.009002 and Player 0 value -0.086393 chips per hand. NashConv sums both players' possible improvement from switching individually to a best response.

One MCCFR iteration performs one sampled traversal for each player, so its iteration count is not directly comparable with full-tree CFR. The 20-million-iteration budget was chosen after 10 million iterations remained above the validation ceiling and the expected inverse-square-root sampling trend supported doubling the work.

The exported policy's 20,000-pair duplicate-deal self-play mean was -0.008525 chips per hand with a 99% confidence interval of [-0.035714, 0.018664], consistent with neutral play.

## Performance

The fixed benchmark used 500,000 iterations, or one million sampled traversals, and five fresh-process repetitions. Only solver training was timed; Numba compilation, startup, exact evaluation, plotting and file writes were excluded.

| Measurement | Reference | Optimised | Change |
|---|---:|---:|---:|
| Median time | 53.2397 s | 1.2479 s | 42.7x faster |
| Typical timing variation (relative MAD) | 0.82% | 1.02% | — |
| Traversals/s | 18,783 | 801,326 | 42.7x higher |
| Peak proportional memory (PSS) | 73.27 MB | 115.58 MB | 57.7% higher |

MAD is the median absolute deviation from the median runtime. PSS divides shared memory among the processes using it. The optimised solver used more memory because the warmed Numba runtime and compiled code remained resident. The fixed benchmark seed measures timing repeatability; strategy quality comes from the separate five-seed evaluation.

Separate 100,000-iteration profiles recorded about 40.9 million Python calls in the recursive reference solver and 85 visible calls in the optimised solver. The compiled traversal performs its internal work below the Python profiler boundary.

## Completion

Seeded tests cover sampling, regret updates, utility perspective, strategy accumulation, alternating player updates and identical-draw implementation agreement. Workflow tests cover resume, snapshot export, checksum validation, registry resolution and playable-agent loading. The convergence, duplicate-deal, benchmark and profiling checks all completed without changing the MCCFR rules.

The playable policy is staged at `artifacts/tabular/leduc-mccfr-final.npz` and registered as `leduc_mccfr_final`.

## Evidence files

- [`validation.json`](validation.json): workload, checks, selected policy metadata and full-precision results.
- [`convergence.csv`](convergence.csv) and [`summary.csv`](summary.csv): per-seed and aggregate convergence.
- [`benchmark_runs.csv`](benchmark_runs.csv) and [`benchmark_summary.csv`](benchmark_summary.csv): raw repetitions and aggregate performance.
- [`gate.json`](gate.json): configuration, environment, evidence index and overall status.
- [`plots/`](plots/): convergence and performance figures.
- [`profiles/`](profiles/): generated reference and optimised CPU profiles.
