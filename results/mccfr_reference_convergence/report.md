# Reference MCCFR convergence on Leduc

**Status: PASS**

## Scope

The reference external-sampling MCCFR solver was trained from five fixed seeds. Each seed used the same canonical Leduc tree and was evaluated independently with the exact best-response evaluator at eight milestones.

Results are in chips per hand. Each player antes 1 chip, creating a 2-chip starting pot; bets are 2 chips in the first round and 4 chips in the second.

One MCCFR iteration performs two sampled root traversals, one for each player. Chance and opponent actions are sampled while every traverser action is explored. The reported strategy is the accumulated average policy, which is the policy used for play and evaluation.

## Workload

- Seeds: `20260810` to `20260814`
- Milestones: 1,000 to 500,000 iterations
- Final work per seed: 500,000 iterations or 1,000,000 sampled player traversals
- Timed region: `solver.train(...)` only, excluding exact evaluation and plotting
- Early stopping: disabled

Five seeds provide a lean check of stochastic variation without turning validation of the deliberately slow reference implementation into a large experiment. The 500,000-iteration endpoint gives a 500-fold span from the first measurement and takes about 33 seconds per seed on this machine.

## Results

| Iterations | Traversals per seed | Median training time | Median exploitability | Seed range |
|---:|---:|---:|---:|---:|
| 1,000 | 2,000 | 0.067 s | 1.28626 | 1.18500 to 1.45015 |
| 5,000 | 10,000 | 0.349 s | 0.50586 | 0.46663 to 0.54924 |
| 10,000 | 20,000 | 0.688 s | 0.30450 | 0.28796 to 0.30980 |
| 25,000 | 50,000 | 1.678 s | 0.17288 | 0.15260 to 0.17521 |
| 50,000 | 100,000 | 3.376 s | 0.10759 | 0.10088 to 0.11379 |
| 100,000 | 200,000 | 6.657 s | 0.07122 | 0.06876 to 0.08256 |
| 250,000 | 500,000 | 16.447 s | 0.04363 | 0.04139 to 0.04832 |
| 500,000 | 1,000,000 | 32.995 s | 0.02985 | 0.02676 to 0.03651 |

Median exploitability decreased at every milestone, and every individual seed improved between the first and final measurements. The median fell by 97.7% across the workload.

The established CFR/CFR+ validation ceiling is 0.005 chips. This MCCFR workload did not reach that final-quality level, but it moved consistently toward it across all seeds. The predeclared MCCFR check required the final median to be no more than ten times that ceiling, or 0.05 chips. This deliberately coarse boundary checks sustained learning without pretending that a short reference run is a final policy. The measured final median was 0.02985 chips.

Raw MCCFR iteration counts must not be compared directly with full-tree CFR iterations because they represent different amounts of work. The plot therefore retains both MCCFR iterations and elapsed solver-training time.

## Files

- `convergence.csv`: exact results for every seed and milestone.
- `summary.csv`: median values and the complete seed range at each milestone.
- `convergence.png`: per-seed and median exploitability against iterations and training time.
- `validation.json`: machine-readable workload, checks, result status, and file descriptions.

Run the validation with:

```bash
python benchmark.py --suite mccfr-reference-convergence
```
