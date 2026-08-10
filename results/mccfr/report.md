# MCCFR validation on Leduc

**Status: PASS**

## Scope

This validation compares the reference and optimised external-sampling MCCFR implementations, checks the optimised solver against exact Leduc strategy evaluation, and verifies that the exported playable policy is neutral against itself under duplicate deals and swapped seats.

Results are in chips per hand. Each player antes 1 chip, creating a 2-chip starting pot; bets are 2 chips in the first round and 4 chips in the second. Lower exploitability is better.

## Workload and rationale

Both implementations use five fixed seeds, `20260810` to `20260814`. At the shared 500,000-iteration workload, the committed reference results are reused rather than needlessly rerun. The optimised solver continues to 20,000,000 iterations per seed.

A 10,000,000-iteration calibration produced about `0.00692` exploitability, above the established `0.005` CFR/CFR+ validation ceiling. MCCFR's usual inverse-square-root sampling trend suggested that doubling the work would reach roughly `0.0049`, making 20,000,000 a proportionate final workload rather than an arbitrary large run. Early stopping was disabled.

One MCCFR iteration performs two sampled traversals, one for each player. MCCFR and full-tree CFR iterations represent very different work and must not be compared directly.

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

## Files

- `convergence.csv`: every exact measurement by implementation, seed, and milestone.
- `summary.csv`: medians and complete seed ranges at each milestone.
- `convergence.png`: exploitability against MCCFR iterations and solver-training time.
- `validation.json`: workload, checks, selected snapshot metadata, and full-precision results.

The playable policy is stored locally at `artifacts/tabular/leduc-mccfr-final.npz` and registered as `leduc_mccfr_final`. The matching resumable checkpoint is under the ignored `runs/` directory. Formal runtime, memory, throughput, and profiler comparisons belong to the next fixed-workload benchmark and are not claimed here.
