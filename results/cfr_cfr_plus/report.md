# CFR/CFR+ correctness and performance gate

**Status: PASS**

## Scope

Reference and optimised CFR/CFR+ use the same indexed Kuhn/Leduc trees, algorithm settings, exact evaluator, iteration counts, and action ordering. One iteration is two full-tree traversals, one for each traversing player.

Measured environment: 8 available CPUs, 15.4 GiB RAM, Python 3.12.3 on `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`.

## Correctness and convergence

Each reference/optimised solver pair is compared after 1 and 3 iterations for both games: 8 comparisons in total. Regrets, strategy sums, current policies, and average policies may differ by no more than 1e-12. **8 of 8 passed.**

| Game | Algorithm | Iterations | Reference exploitability | Optimised exploitability | Absolute gap | Result |
|---|---:|---:|---:|---:|---:|---:|
| Kuhn | CFR | 10,000 | 0.00011101215 | 0.00011101215 | 0 | PASS |
| Kuhn | CFR+ | 10,000 | 9.0495148e-06 | 9.0495148e-06 | 0 | PASS |
| Leduc | CFR | 5,000 | 0.0032213533 | 0.0033375035 | 0.00011615029 | PASS |
| Leduc | CFR+ | 5,000 | 1.3825689e-05 | 1.7358688e-05 | 3.5329985e-06 | PASS |

Kuhn poker has a known equilibrium value: Player 0 expects to lose 1/18 of a chip per hand, about 0.05556 chips, when both players use equilibrium strategies. The measured value is checked against this result. Leduc uses exact best-response exploitability and cross-implementation agreement because no independent equilibrium value is used here.

Kuhn must reach at most 0.001-chip exploitability and 0.0001-chip value error. Leduc must reach at most 0.005-chip exploitability. Reference/optimised exploitability gaps are limited to 0.0002 chip on Kuhn and 0.001 chip on Leduc, one fifth of their quality ceilings. These thresholds were fixed before this run.

| Game | Algorithm | Reference P0 value | Optimised P0 value | Reference NashConv | Optimised NashConv |
|---|---:|---:|---:|---:|---:|
| Kuhn | CFR | -0.05555240 | -0.05555240 | 0.0002220243 | 0.0002220243 |
| Kuhn | CFR+ | -0.05555556 | -0.05555556 | 1.809903e-05 | 1.809903e-05 |
| Leduc | CFR | -0.08577175 | -0.08578214 | 0.0064427065 | 0.0066750071 |
| Leduc | CFR+ | -0.08560592 | -0.08560593 | 2.7651378e-05 | 3.4717375e-05 |

P0 value is player zero's expected chip result under the average strategies. NashConv is the sum of both players' possible best-response improvements; exploitability is half of NashConv here.

## Duplicate-deal self-play

Each sampled physical deal is played twice with the same frozen optimised policy. The focal copy occupies Player 0 first and Player 1 in the replay. A check passes when zero lies inside the paired mean's predeclared 99% confidence interval.

| Game | Algorithm | Duplicate pairs | Mean chips | 99% confidence interval | Result |
|---|---:|---:|---:|---:|---:|
| Kuhn | CFR | 20,000 | -0.0012 | [-0.00866451, 0.00626451] | PASS |
| Kuhn | CFR+ | 20,000 | -0.001675 | [-0.00927355, 0.00592355] | PASS |
| Leduc | CFR | 20,000 | 0.016575 | [-0.0102394, 0.0433894] | PASS |
| Leduc | CFR+ | 20,000 | -0.0164 | [-0.0430198, 0.0102198] | PASS |

## Fixed-workload performance

| Game | Algorithm | Iterations | Reference median | Reference relative MAD | Optimised median | Optimised relative MAD | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kuhn | CFR | 10,000 | 1.9494 s | 0.9% | 0.0713 s | 1.9% | 27.3x |
| Kuhn | CFR+ | 10,000 | 2.0305 s | 1.2% | 0.0739 s | 8.0% | 27.5x |
| Leduc | CFR | 5,000 | 121.5935 s | 1.4% | 0.9226 s | 5.2% | 131.8x |
| Leduc | CFR+ | 5,000 | 118.9734 s | 1.3% | 0.8601 s | 1.2% | 138.3x |

| Game | Algorithm | Reference traversals/s | Optimised traversals/s | Reference peak PSS | Optimised peak PSS | Memory change |
|---|---:|---:|---:|---:|---:|---:|
| Kuhn | CFR | 10260 | 280312 | 68.0 MB | 100.4 MB | +47.6% |
| Kuhn | CFR+ | 9850 | 270461 | 68.0 MB | 100.4 MB | +47.7% |
| Leduc | CFR | 82 | 10839 | 72.9 MB | 101.3 MB | +38.9% |
| Leduc | CFR+ | 84 | 11626 | 72.9 MB | 101.3 MB | +38.9% |

Timing uses the median of 5 fresh-process runs. Relative MAD is the median absolute deviation divided by the median, a robust percentage measure of variation. Absolute MAD values and every repeat remain in the CSV files. Numba compilation, exact evaluation, plotting, process startup, and file writes are outside the timer.

Peak memory is process-tree PSS, sampled every 10 ms during training. The optimised solvers use more process memory here because the measurement includes the resident Numba runtime after mandatory JIT warm-up.

CFR and CFR+ are deterministic here, so repeated runs measure timing and memory variation rather than different learning seeds. Early stopping is disabled.

## Workload rationale

Kuhn uses 10,000 iterations and Leduc uses 5,000 for timing. This keeps the optimised Leduc timed region around one second while five fresh-process reference repeats remain feasible. Convergence is checked separately through 10,000 Kuhn and 5,000 Leduc iterations, so timing scale is not mistaken for the strategy-quality acceptance scale. CFR+ uses a fixed averaging delay of 10 iterations in both implementations.

The iteration plots compare strategy quality after equal algorithmic work. The reference and optimised lines closely overlap because they perform the same CFR updates. The time plots use independently measured cumulative training time, so horizontal separation shows the execution-speed difference.

## Profiler evidence

Separate `cProfile` runs use Leduc and 100 iterations. Profiling changes runtime, so these runs are excluded from formal timings. The generated Markdown files in `profiles/` contain the top 25 functions by cumulative time.

`cProfile` counts Python calls. The reference solver makes millions of visible recursive, list, generator, enum, and dictionary calls. The optimised solver moves the same tree calculations inside a Numba-compiled machine-code kernel, so the profiler sees only calls into that kernel rather than its internal operations. The much smaller optimised call count therefore does not mean less poker work was done.

## Files

- `gate.json`: machine-readable configuration, definitions, environment, checks, artefact descriptions, and pass/fail status.
- `convergence.csv`: exact quality measurements at every declared milestone.
- `benchmark_runs.csv`: every raw timing and peak-memory repetition.
- `benchmark_summary.csv`: medians, variation, throughput, memory, and final quality.
- `plots/`: convergence and engineering-comparison figures.
- `profiles/`: automatically generated reference and optimised CPU profiles.
