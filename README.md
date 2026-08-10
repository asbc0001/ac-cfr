# AC CFR

A research project for implementing, validating, and comparing counterfactual regret minimisation algorithms for two-player imperfect-information poker games. It currently provides complete reference and optimised CFR and CFR+ pipelines for Kuhn and Leduc poker, together with the shared Hold'em foundations required by later Deep CFR work.

## Current functionality

- Complete Kuhn and Leduc engines with reference and optimised CFR/CFR+ pipelines.
- Exact evaluation, checkpointed training, snapshots, plotting, and reproducible benchmarks.
- Validated tabular agents and a checksum-protected strategy registry.
- Conventional and modified HULHE engines with compact cards, fast evaluation, and suit-canonical information states.

## CFR and CFR+

An information set groups decision states that the acting player cannot distinguish using the information available to them. At each information set, CFR compares the value of the strategy it used with the value it would have received by choosing each legal action instead:

```text
regret(I, a) += counterfactual reach × (action value(I, a) - strategy value(I))
```

Counterfactual reach is the probability that chance and the opponent reach the decision. It excludes the updating player's probability, allowing every legal action to be evaluated as though that player had reached the decision. Regret matching normalises positive cumulative regrets into action probabilities, using a uniform strategy if none are positive.

Each iteration updates Player 0 and then Player 1 through separate full-tree traversals with a policy frozen for each pass. Accumulated reach-weighted strategies form the average policy used for evaluation and play.

CFR+ keeps the same traversal but clips updated cumulative regrets to zero and weights later average-policy contributions linearly after a configurable delay. This usually produces a strong average strategy in fewer iterations without changing the game or evaluation rules.

## Implementations and optimisation

Recursive Python solvers provide an independent correctness baseline. The optimised solvers preserve their update order and mathematics while using:

- precomputed dense indexed trees with stable node, information-set, child, depth, and action indices;
- compact flat NumPy arrays instead of nested Python objects and dictionaries in the hot path;
- reusable reach, value, regret, and policy buffers to avoid repeated allocation;
- forward reach propagation and reverse node-value passes over deterministic array order; and
- a cached Numba-compiled training kernel that moves the repeated numerical loops into machine code.

NumPy provides the arrays, while Numba compiles the training kernel into machine code. Benchmarks warm this compilation separately so it is not counted as solver training.

## Exact evaluation

Kuhn and Leduc are evaluated without sampling. The evaluator traverses every chance outcome and action probability, then finds each best response while enforcing one action across each information set.

All Kuhn and Leduc values use the games' base chip unit and are reported per hand. Each player antes 1 chip, so both games begin with a 2-chip pot. A Kuhn bet is 1 chip; Leduc bets are 2 chips in the first round and 4 chips in the second round.

`NashConv` sums both players' possible improvement from switching individually to a best response. Exploitability is reported as `NashConv / 2`, so lower is better. At Kuhn equilibrium, Player 0 loses `1/18` of the 1-chip ante per hand, approximately `0.05556` chips or 5.56% of one ante. Leduc is validated through exact exploitability and agreement between independently structured solvers.

## Validation and results

The optimised solvers matched reference regrets, strategy sums, and policies to an absolute tolerance of `1e-12` in all eight deterministic comparisons. Longer convergence checks and duplicate-deal, swapped-seat self-play also passed.

| Game | Policy | Final training iterations | Final exploitability | Reference-to-optimised speedup |
|---|---|---:|---:|---:|
| Kuhn | CFR | 100,000 | 0.00001719 | 27.3x |
| Kuhn | CFR+ | 100,000 | 0.00000215 | 27.5x |
| Leduc | CFR | 500,000 | 0.00014595 | 131.8x |
| Leduc | CFR+ | 500,000 | 0.00000017 | 138.3x |

Final policies use larger budgets than benchmarks. Kuhn uses ten times its already near-target correctness run. Leduc uses 100 times its correctness run, chosen from ordinary CFR's inverse-square-root convergence to pass the `0.0005` target. CFR+ uses the same budgets.

Benchmarks use 10,000 Kuhn and 5,000 Leduc iterations with five fresh-process repetitions. After Numba warm-up they measure median training time, variation, and peak process-tree memory. The optimised solvers are much faster but use roughly 39% to 48% more memory because the Numba runtime remains resident.

Detailed configurations, full-precision tables, plots, raw repetitions, memory results, and profiler output are available in:

- [CFR/CFR+ correctness and performance results](results/cfr_cfr_plus/report.md)
- [Final tabular policy results](results/tabular_policies/report.md)

## Training and policy artefacts

Final-policy production trains to a fixed budget, exports average-policy snapshots, evaluates them exactly, stages the selected snapshot under `artifacts/`, and records its compatibility data and checksum in the registry. `TabularAgent` exposes the frozen policy without depending on the trainer.

Start a training run with an explicit budget and snapshot milestones:

```bash
python train.py \
    --game leduc \
    --solver cfr_plus \
    --iterations 500000 \
    --run-id leduc-cfr-plus-final \
    --evaluation-interval 5000 \
    --checkpoint-interval 50000 \
    --snapshot-iterations 50000,200000,500000 \
    --averaging-delay 10 \
    --plot
```

Resume the exact saved run configuration and state with:

```bash
python train.py --resume runs/leduc-cfr-plus-final/checkpoints/latest.npz
```

A checkpoint contains the state needed to resume training. A smaller strategy snapshot contains only the normalised average policy and compatibility metadata needed for evaluation or play.

Snapshots use non-executable NumPy data loaded with `allow_pickle=False`. Before constructing a `TabularAgent`, the registry verifies a trusted project-relative path, file size, SHA-256 checksum, game, encoding, action space, schema, and game-tree digest.

After the registered snapshot has been placed under its declared `artifacts/` path, evaluate it with:

```bash
python evaluate.py leduc_cfr_plus_final
```

Plot one run or compare several runs with:

```bash
python plot_results.py runs/leduc-cfr-final runs/leduc-cfr-plus-final
```

Run output stays under ignored `runs/`, selected snapshots under ignored `artifacts/`, and compact evidence under version-controlled `results/`. Policy publication and downloading will be added with the web work.

## Modified HULHE

Modified HULHE begins on the flop with each player contributing one small bet to a two-small-bet pot. Flop, turn, and river use standard heads-up fixed-limit position and sizing, but each round allows only an opening bet and one raise.

The same engine supports conventional HULHE from the pre-flop blinds with four betting levels per round. Modified HULHE is the planned Deep CFR target.

## Setup

Python 3.12 or later is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Checks

Run the normal development checks before committing:

```bash
ruff check .
ruff format --check .
pyright
pytest
```

The default test suite excludes expensive tests marked as slow. Run the full evaluator checks separately when changing evaluator code or lookup data:

```bash
pytest -m slow
```

CI also verifies packaged evaluator data and command-line entry points outside the checkout.

## Architecture

```text
src/ac_cfr/
├── agents/          Frozen playable policies and baseline agents
├── benchmarking/    Repeated timing, memory, profiling, and correctness gates
├── common/          Configuration identifiers and deterministic RNG handling
├── evaluation/      Exact best responses, metrics, self-play, and plotting
├── games/           Kuhn, Leduc, Hold'em, and shared indexed-tree contracts
├── persistence/     Checkpoints, snapshots, registry, and compact results
├── solvers/         Reference and optimised CFR and CFR+
└── training/        Reproducible training schedules and metric recording
```

Game states hold complete hand data, while agents and solvers receive only player-visible `InformationState` values. Kuhn and Leduc use precomputed trees; Hold'em uses compact on-demand transitions rather than pre-enumerating its full tree.

The fast Hold'em evaluator uses reproducible packaged lookup tables validated against independent reference implementations. Regenerate them with:

```bash
python tools/generate_holdem_evaluator.py
```

## Repository conventions

Development uses a single `main` branch. Run the local checks before each direct commit or push; CI verifies every push to `main`.

Compact evidence belongs under `results/`, downloaded strategies under ignored `artifacts/`, and training output under ignored `runs/`. Small deterministic evaluator tables are committed, but generated policies and models remain outside Git history.

Next: implement and validate external-sampling MCCFR on Leduc using the same training, evaluation, snapshot, and benchmarking foundations.
