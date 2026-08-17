# CFR and CFR+ on Kuhn and Leduc

This directory contains correctness, convergence, self-play, profiling and fixed-workload performance evidence for the reference and optimised CFR/CFR+ implementations.

**Status: PASS**

## Validation

Reference and optimised solvers used the same indexed trees, algorithm settings, exact evaluator, iteration counts and action ordering. One iteration performs one full-tree traversal for each player.

Values are chips per hand. Both games start from a two-chip ante pot; Kuhn bets are one chip, while Leduc bets are two chips in the first round and four in the second.

Regrets, strategy sums, current policies and average policies were compared after one and three iterations for both algorithms and games. All eight comparisons agreed within `1e-12`.

| Game | Algorithm | Iterations | Reference exploitability | Optimised exploitability | Absolute gap | Result |
|---|---|---:|---:|---:|---:|---|
| Kuhn | CFR | 10,000 | 0.00011101215 | 0.00011101215 | 0 | PASS |
| Kuhn | CFR+ | 10,000 | 9.0495148e-06 | 9.0495148e-06 | 0 | PASS |
| Leduc | CFR | 5,000 | 0.0032213533 | 0.0033375035 | 0.00011615029 | PASS |
| Leduc | CFR+ | 5,000 | 1.3825689e-05 | 1.7358688e-05 | 3.5329985e-06 | PASS |

Kuhn had to reach at most 0.001 exploitability and 0.0001 Player 0 value error against its known equilibrium value of `-1/18` chips. Leduc had to reach at most 0.005 exploitability. The reference/optimised gaps were limited to 0.0002 chips on Kuhn and 0.001 chips on Leduc; all checks passed.

| Game | Algorithm | Reference P0 value | Optimised P0 value | Reference NashConv | Optimised NashConv |
|---|---|---:|---:|---:|---:|
| Kuhn | CFR | -0.05555240 | -0.05555240 | 0.0002220243 | 0.0002220243 |
| Kuhn | CFR+ | -0.05555556 | -0.05555556 | 1.809903e-05 | 1.809903e-05 |
| Leduc | CFR | -0.08577175 | -0.08578214 | 0.0064427065 | 0.0066750071 |
| Leduc | CFR+ | -0.08560592 | -0.08560593 | 2.7651378e-05 | 3.4717375e-05 |

P0 value is Player 0's expected chip result. NashConv sums both players' possible improvement from switching individually to a best response; exploitability is half of NashConv here.

## Duplicate-deal self-play

Each physical deal was played twice with the same saved optimised policy and swapped seats. Every planned 99% confidence interval contained zero.

| Game | Algorithm | Duplicate pairs | Mean chips | 99% confidence interval |
|---|---|---:|---:|---:|
| Kuhn | CFR | 20,000 | -0.0012 | [-0.00866451, 0.00626451] |
| Kuhn | CFR+ | 20,000 | -0.001675 | [-0.00927355, 0.00592355] |
| Leduc | CFR | 20,000 | 0.016575 | [-0.0102394, 0.0433894] |
| Leduc | CFR+ | 20,000 | -0.0164 | [-0.0430198, 0.0102198] |

## Performance

Five fresh-process repetitions timed only solver training after mandatory Numba warm-up. Exact evaluation, plotting, startup and file writes were excluded.

| Game | Algorithm | Iterations | Reference median | Optimised median | Speedup |
|---|---|---:|---:|---:|---:|
| Kuhn | CFR | 10,000 | 1.9494 s | 0.0713 s | 27.3x |
| Kuhn | CFR+ | 10,000 | 2.0305 s | 0.0739 s | 27.5x |
| Leduc | CFR | 5,000 | 121.5935 s | 0.9226 s | 131.8x |
| Leduc | CFR+ | 5,000 | 118.9734 s | 0.8601 s | 138.3x |

| Game | Algorithm | Reference traversals/s | Optimised traversals/s | Reference peak memory | Optimised peak memory |
|---|---|---:|---:|---:|---:|
| Kuhn | CFR | 10,260 | 280,312 | 68.0 MB | 100.4 MB |
| Kuhn | CFR+ | 9,850 | 270,461 | 68.0 MB | 100.4 MB |
| Leduc | CFR | 82 | 10,839 | 72.9 MB | 101.3 MB |
| Leduc | CFR+ | 84 | 11,626 | 72.9 MB | 101.3 MB |

Memory is process-tree proportional set size (PSS), which divides shared memory among the processes using it. The optimised solvers used more because the warmed Numba runtime and compiled code remained resident. Timing uses 10,000 Kuhn and 5,000 Leduc iterations to keep the optimised Leduc measurement near one second while retaining practical reference repeats. CFR+ used an averaging delay of 10 in both implementations.

Separate 100-iteration Leduc profiles supplement rather than contribute to the formal timings. They show the reference solvers spending most of their time in recursive Python traversal, while the optimised solvers move the same tree calculations into compiled kernels.

## Evidence files

- [`gate.json`](gate.json): configuration, environment, checks and overall status.
- [`convergence.csv`](convergence.csv): exact quality measurements at each recorded iteration.
- [`benchmark_runs.csv`](benchmark_runs.csv) and [`benchmark_summary.csv`](benchmark_summary.csv): raw repetitions and aggregate performance.
- [`plots/`](plots/): convergence and performance figures.
- [`profiles/`](profiles/): generated reference and optimised CPU profiles.
