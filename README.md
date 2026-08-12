# AC CFR

A research project for implementing, validating, and comparing counterfactual regret minimisation algorithms for two-player imperfect-information poker. It includes reference and optimised CFR, CFR+, external-sampling MCCFR, and Deep CFR pipelines, plus a playable web demo.

## Current functionality

- Complete Kuhn and Leduc engines with reference and optimised tabular solvers, plus Leduc Deep CFR.
- Exact evaluation, resumable training, playable snapshots, plotting, and reproducible benchmarks.
- Validated tabular and neural agents with a checksum-protected strategy registry.
- Conventional and modified HULHE engines with compact cards, fast evaluation, and suit-canonical information states.
- An ephemeral FastAPI and vanilla-JavaScript interface for playing against frozen policies.

## CFR and CFR+

An information set groups decision states that the acting player cannot distinguish using the information available to them. At each information set, CFR compares the value of the strategy it used with the value it would have received by choosing each legal action instead:

```text
regret(I, a) += counterfactual reach × (action value(I, a) - strategy value(I))
```

Counterfactual reach is the probability that chance and the opponent reach the decision. It excludes the updating player's probability, allowing every legal action to be evaluated as though that player had reached the decision. Regret matching normalises positive cumulative regrets into action probabilities, using a uniform strategy if none are positive.

Each iteration updates Player 0 and then Player 1 through separate full-tree traversals with a policy frozen for each pass. Accumulated reach-weighted strategies form the average policy used for evaluation and play.

CFR+ keeps the same traversal but clips updated cumulative regrets to zero and weights later average-policy contributions linearly after a configurable delay. This usually produces a strong average strategy in fewer iterations without changing the game or evaluation rules.

## External-sampling MCCFR

Full-tree CFR visits every chance and action branch on each traversal. External-sampling MCCFR reduces that work by sampling one chance outcome and one opponent action at each relevant decision while still exploring every action available to the player whose regrets are being updated.

Each outer iteration performs one sampled traversal for Player 0 and one for Player 1. Regrets are updated after each traversal, so the next traversal sees the latest strategy. Over many iterations, the sampled paths estimate the full update correctly on average, but individual runs are noisy. Strategy quality is therefore evaluated across several fixed seeds rather than from one favourable run.

One MCCFR iteration is much cheaper than one full-tree CFR iteration because it visits only part of the tree. Raw iteration counts therefore cannot rank CFR and MCCFR directly. Convergence over training time and fixed-workload throughput provide more meaningful comparisons.

## Deep CFR

Deep CFR keeps MCCFR's sampled traversal but replaces regret and average-strategy tables with neural networks. Each player's advantage network predicts the relative value of every action; regret matching turns positive predictions into the current strategy, falling back to the highest prediction if none are positive. A separate shared strategy network learns the strategy history and becomes the playable average policy.

Each outer iteration gives both players `K` sampled traversals. Their targets enter bounded uniform reservoirs, where every sample seen has an equal chance of being retained. Fresh advantage networks train for fixed numbers of sampled minibatch updates, weighted by generating iteration under Linear CFR. Fixed update budgets stop neural-training cost growing automatically with reservoir size. Exported strategy networks train separately from the shared strategy reservoir and never feed back into traversal.

Leduc uses a versioned 37-value encoding of the acting player, cards, betting round, and action history. The selected network has three 64-unit ReLU hidden layers and three masked action outputs. Held-out losses, finite-value checks, and gradient clipping guard training; dropout remained disabled because validation showed no benefit.

The final preset uses 1,000 traversals and 1,000 advantage updates per player per iteration, 512-sample batches, and 100,000-sample reservoirs. The complete configuration is in [`configs/deep_cfr/leduc_final.toml`](configs/deep_cfr/leduc_final.toml).

## Implementations and optimisation

Recursive Python tabular solvers provide an independent correctness baseline. Their optimised counterparts preserve the update order and mathematics while using:

- precomputed dense indexed trees with stable node, information-set, child, depth, and action indices;
- compact flat NumPy arrays instead of nested Python objects and dictionaries in the hot path;
- reusable value, regret, policy, sampling, and traversal buffers to avoid repeated allocation;
- forward reach propagation and reverse node-value passes for full-tree CFR/CFR+;
- compact sampled-action arrays and reusable arrays that record pending tree nodes instead of recursive Python calls for MCCFR; and
- a cached Numba-compiled training kernel that moves the repeated numerical loops into machine code.

NumPy provides the arrays, while Numba compiles the training kernel into machine code. Benchmarks warm this compilation separately so it is not counted as solver training.

Optimised Deep CFR batches network inference across pending traversal states, stores reservoir samples in packed arrays, and replaces recursive Python traversal bookkeeping with reusable indexed buffers. These changes reduce Python-visible calls and network invocations while preserving the algorithmic work and learning semantics used by the matched reference comparison.

## Exact evaluation

Kuhn and Leduc are evaluated without sampling. The evaluator traverses every chance outcome and action probability, then finds each best response while enforcing one action across each information set.

All Kuhn and Leduc values use the games' base chip unit and are reported per hand. Each player antes 1 chip, so both games begin with a 2-chip pot. A Kuhn bet is 1 chip; Leduc bets are 2 chips in the first round and 4 chips in the second round.

`NashConv` sums both players' possible improvement from switching individually to a best response. Exploitability is reported as `NashConv / 2`, so lower is better. At Kuhn equilibrium, Player 0 loses `1/18` of the 1-chip ante per hand, approximately `0.05556` chips or 5.56% of one ante. Leduc is validated through exact exploitability and agreement between independently structured solvers.

## Validation and results

The optimised CFR/CFR+ solvers matched reference regrets, strategy sums, and policies to an absolute tolerance of `1e-12` in all eight deterministic comparisons. MCCFR also matched its reference updates within `1e-12` when given identical sampled draws. Longer convergence checks and duplicate-deal, swapped-seat self-play passed for all three algorithms.

| Game | Policy | Training budget / selected snapshot | Final exploitability | Fixed-benchmark speedup |
|---|---|---:|---:|---:|
| Kuhn | CFR | 100,000 | 0.00001719 | 27.3x |
| Kuhn | CFR+ | 100,000 | 0.00000215 | 27.5x |
| Leduc | CFR | 500,000 | 0.00014595 | 131.8x |
| Leduc | CFR+ | 500,000 | 0.00000017 | 138.3x |
| Leduc | MCCFR | 20,000,000 | 0.00450098 | 42.7x |
| Leduc | Deep CFR | 200; selected 150 | 0.20564932 | 6.55x |

Final CFR/CFR+ policies use larger budgets than their validation and benchmark workloads. Kuhn policies train for 100,000 iterations rather than 10,000. Leduc policies train for 500,000 rather than 5,000, with the larger budget chosen from CFR's measured convergence to pass the `0.0005` final-policy target.

MCCFR was trained to 20,000,000 iterations across five seeds. Median exact exploitability reached `0.004501`, below the established `0.005` validation ceiling. The seed with the median final result was selected rather than the best seed, avoiding a cherry-picked final policy. That policy is registered as `leduc_mccfr_final`.

The final optimised Deep CFR run completed 200 outer iterations and 400,000 sampled traversals. Exact evaluation selected iteration 150 at `0.205649` exploitability instead of the weaker final snapshot. Iterations 20, 75, and 150 are registered as early, intermediate, and final policies. This validates the neural training and playable-policy lifecycle; it does not claim parity with the stronger tabular policies.

CFR/CFR+ benchmarks use 10,000 Kuhn and 5,000 Leduc iterations. The MCCFR benchmark uses 500,000 Leduc iterations, or 1,000,000 sampled traversals, giving the optimised workload more than one second of measured training. These tabular benchmarks use five fresh-process repetitions after Numba warm-up and record median runtime, variation, traversals per second, and peak process-tree memory.

Optimised MCCFR completed its fixed workload in `1.248` seconds, compared with `53.240` seconds for the reference implementation, a `42.7x` speedup. Separate 100,000-iteration profiles show that the reference solver spends most of its time in recursive Python traversal, sampling, and policy normalisation. The optimised traversal runs inside the compiled Numba kernel, whose internal operations are not visible to Python's profiler.

Across three matched repetitions, optimised Deep CFR took a median `1.48` seconds versus `9.72` seconds for the reference implementation: `6.55x` faster with `15.4%` lower peak process-tree memory. Profiling reduced network calls from 30,315 to 365 and Python-visible calls from 19.3 million to 1.5 million. Single-process collection reached `7,997.6` traversals/second; the full final run averaged `519.4` once neural training and snapshot exports were included. These Leduc measurements are not a multi-process or modified-HULHE forecast.

Detailed configurations, full-precision tables, plots, raw repetitions, memory results, and profiler output are available in:

- [CFR/CFR+ correctness and performance results](results/cfr_cfr_plus/README.md)
- [MCCFR validation and performance results](results/mccfr/README.md)
- [Deep CFR validation, performance, and final-policy results](results/deep_cfr/README.md)
- [Modified-HULHE cloud calibration and worker-scaling results](results/modified_hulhe/README.md)
- [Final tabular policy results](results/tabular_policies/README.md)

## Training and policy artefacts

To produce a final policy, training runs to a fixed budget, exports average-policy snapshots, evaluates them, places the selected snapshot under `artifacts/`, and records its compatibility data and checksum in the registry. `TabularAgent` and `NeuralAgent` then expose frozen policies without depending on the trainer.

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

The same persistence infrastructure supports MCCFR. Its checkpoint also saves the chance and policy random-number generator states, so a resumed run continues with the same future samples.

Deep CFR runs save their resolved TOML configuration. Checkpoints retain the networks, reservoirs, random-number-generator states, milestones, metrics, elapsed time, and architecture metadata needed for compatible resume. Smaller playable snapshots contain only the frozen average-strategy network and reconstruction metadata.

Start or resume the final Leduc Deep CFR configuration with:

```bash
python train.py \
    --config configs/deep_cfr/leduc_final.toml \
    --run-id leduc-deep-cfr-final

python train.py --resume runs/leduc-deep-cfr-final/checkpoints/latest.pt
```

Tabular snapshots use non-executable NumPy data loaded with `allow_pickle=False`. Deep CFR snapshots load only expected PyTorch weights, reconstruct a named architecture, disable gradients, mask illegal actions, and reject incompatible metadata or tensor shapes. Before constructing either agent, the registry verifies a trusted project-relative path, file size, SHA-256 checksum, game, encoding, action space, schema, and applicable tree or model compatibility identifiers.

After the registered snapshot has been placed under its declared `artifacts/` path, evaluate it with:

```bash
python evaluate.py leduc_cfr_plus_final
python evaluate.py leduc_mccfr_final
python evaluate.py leduc_deep_cfr_final
```

Plot one run or compare several runs with:

```bash
python plot_results.py runs/leduc-cfr-final runs/leduc-cfr-plus-final
```

Run output stays under ignored `runs/`, selected snapshots under ignored `artifacts/`, and compact evidence under version-controlled `results/`.

## Web demo

The single-page FastAPI demo plays directly through the shared game engines and frozen `PlayableAgent` implementations. It never trains a solver during a request.

The current registry exposes:

| Game | Playable opponents |
|---|---|
| Kuhn | Random, final CFR, final CFR+ |
| Leduc | Random, final CFR/CFR+/MCCFR, early/intermediate/final Deep CFR |
| Modified HULHE | Random, rule-based, temporary local Deep CFR development snapshot |

Install registry-declared strategy snapshots from a staged release directory with:

```bash
python download_models.py --source-directory /path/to/release-assets
```

`ac-cfr-download-models` is the equivalent installed command. Without `--source-directory`, it fetches assets from the GitHub release tags recorded in the registry. Installation is atomic and requires the declared file size and SHA-256 checksum to match. Random and rule-based opponents require no downloaded file.

The two-iteration modified-HULHE snapshot exists only for local interface development and provides no strategy-quality evidence. Its registry entry and artefact will be replaced after the final cloud policy is selected.

Start the local single-worker application with:

```bash
ac-cfr-web --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`. The selected game's concise rules are available from the table setup panel. The browser receives only player-visible state. Hands live temporarily in one server process, use opaque random identifiers and version-checked actions, and are not stored in cookies, browser storage or a database. Refreshing loses the current hand; **New hand** explicitly discards it before dealing another.

Build and run the same application in Docker with:

```bash
docker build -t ac-cfr-web .
docker run --rm --user "$(id -u):$(id -g)" -p 8000:8000 \
    -v "$(pwd)/artifacts:/app/artifacts:ro" \
    ac-cfr-web
```

The user mapping lets the non-root container read owner-only local snapshots without weakening their permissions. The image uses CPU-only PyTorch because the demo performs inference on CPU; cloud training retains its CUDA environment. The artefact mount is optional when using only random or rule-based opponents. The application uses one Uvicorn worker because hands are held temporarily in process memory.

## Modified HULHE

Modified HULHE begins on the flop with each player contributing one small bet to a two-small-bet pot. Flop, turn, and river use standard heads-up fixed-limit position and sizing, but each round allows only an opening bet and one raise.

The same engine supports conventional HULHE from the pre-flop blinds with four betting levels per round. Modified HULHE is the production Deep CFR target.

Cloud presets declare `storage_budget_bytes` as the usable persistent-storage ceiling for the run. Preflight uses the smaller of this configured budget and the backing filesystem's reported free space. Live metrics and checkpoint guards also subtract existing run files from the configured budget. The separate backing-filesystem value remains visible because shared filesystems may report capacity that is not allocated to the current machine. Memory checks use the effective cgroup limit rather than the host's physical-memory total.

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
├── solvers/         Reference and optimised CFR, CFR+, MCCFR, and Deep CFR
├── training/        Reproducible tabular and neural training schedules
└── web/             Ephemeral FastAPI gameplay and packaged browser assets
```

Game states hold complete hand data, while playable agents act through player-visible `InformationState` values. Solver strategies are also indexed by information set, preventing decisions from using hidden opponent cards. Kuhn and Leduc use precomputed trees; Hold'em uses compact on-demand transitions rather than pre-enumerating its full tree.

The fast Hold'em evaluator uses reproducible packaged lookup tables validated against independent reference implementations. Regenerate them with:

```bash
python tools/generate_holdem_evaluator.py
```

## Repository conventions

Development uses a single `main` branch. Run the local checks before each direct commit or push; CI verifies every push to `main`.

Compact evidence belongs under `results/`, playable strategy snapshots under ignored `artifacts/`, and training output under ignored `runs/`. Small deterministic evaluator tables are committed, but generated policies and models remain outside Git history.

Next: select and publish the final modified-HULHE policy, replace the temporary registry entry, repeat the completed container gate with that final artefact, and deploy the demo.
