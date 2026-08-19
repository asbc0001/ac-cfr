# Usage and development

Run commands from the repository root. Python 3.12 or later is required.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Installation provides the `ac-cfr-*` commands used below.

## Released policies

Policy files are distributed separately from the repository. Install every file declared by the
strategy registry:

```bash
ac-cfr-download-models
```

Install selected policies by repeating `--strategy-id`:

```bash
ac-cfr-download-models \
    --strategy-id leduc_cfr_plus_final \
    --strategy-id leduc_deep_cfr_final
```

Files can instead be installed from a local release staging directory:

```bash
ac-cfr-download-models --source-directory /path/to/release-assets
```

The installer writes to `artifacts/` and verifies each file against the size and SHA-256 checksum
in [`configs/strategy_registry.json`](../configs/strategy_registry.json). Random and rule-based
policies have no associated files.

## Evaluation

### Kuhn and Leduc

Registered Kuhn and Leduc policies support exact expected-value, NashConv and exploitability
evaluation. Keep ad hoc output under `runs/` rather than modifying curated evidence:

```bash
ac-cfr-evaluate leduc_cfr_plus_final \
    --results runs/local-evaluations.csv

ac-cfr-evaluate leduc_deep_cfr_final \
    --results runs/local-evaluations.csv
```

### Modified HULHE

Modified HULHE is evaluated through duplicate-deal matches rather than exact exploitability. The
following runs the final snapshot against the rule-based policy:

```bash
ac-cfr-evaluate modified-hulhe \
    --snapshot artifacts/deep_cfr/modified-hulhe-deep-cfr-final.pt \
    --include-rule-based \
    --duplicate-pairs 1000 \
    --seed 20260811 \
    --confidence-level 0.95 \
    --bootstrap-resamples 1000 \
    --results runs/modified-hulhe-evaluation.csv
```

Repeat `--snapshot` for a progression round-robin, or use `--anchor-snapshot` for fixed snapshot
opponents. Larger duplicate-pair and bootstrap counts reduce sampling variation at additional
runtime cost. The resulting measurements are opponent-specific and are not exploitability
estimates.

## Local web application

Install the released policies, then start the single-process application:

```bash
ac-cfr-web --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Hands are held in process memory and are lost on refresh or server
restart.

To run the production image locally:

```bash
docker build -t ac-cfr-web:local .
docker run --rm -p 8000:8000 ac-cfr-web:local
```

The build downloads and verifies the registered policies. The final image runs as a non-root
user with CPU-only PyTorch and listens on port 8000.

## Training

### Run output

Each job writes to `runs/<run-id>/`:

```text
runs/<run-id>/
├── run_config.json
├── metrics.csv
├── summary.txt
├── checkpoints/
└── strategy_snapshots/
```

Checkpoints contain the complete state needed for compatible resume. Strategy snapshots contain
the average policy and reconstruction metadata used for evaluation and play. Training handles
`Ctrl+C` and termination requests at iteration boundaries; resume from the reported `latest`
checkpoint.

### Tabular CFR, CFR+ and MCCFR

Example Kuhn CFR+ run:

```bash
ac-cfr-train \
    --game kuhn \
    --solver cfr_plus \
    --iterations 10000 \
    --run-id example-kuhn-cfr-plus \
    --evaluation-interval 1000 \
    --checkpoint-interval 2500 \
    --snapshot-iterations 1000,5000,10000 \
    --averaging-delay 10 \
    --plot
```

The optimised `cfr` and `cfr_plus` solvers support Kuhn and Leduc; `mccfr` supports Leduc. The
corresponding `naive_cfr`, `naive_cfr_plus` and `naive_mccfr` implementations are intended for
correctness comparisons and are substantially slower.

Resume a tabular run:

```bash
ac-cfr-train \
    --resume runs/example-kuhn-cfr-plus/checkpoints/latest.npz
```

Tabular resume uses the original configuration and total iteration budget. New-run options cannot
be supplied alongside `--resume`.

### Deep CFR on Leduc

New Deep CFR runs require a TOML preset:

```bash
ac-cfr-train \
    --config configs/deep_cfr/leduc_final.toml \
    --run-id example-leduc-deep-cfr \
    --plot
```

`leduc_final.toml` is the selected 200-iteration workload, not a smoke test. Deep CFR command-line
options can override preset values, although separate TOML files are preferable for experiments
that need an auditable configuration.

Resume from the latest checkpoint:

```bash
ac-cfr-train \
    --resume runs/example-leduc-deep-cfr/checkpoints/latest.pt
```

Deep CFR resume can extend the target iteration count:

```bash
ac-cfr-train \
    --resume runs/example-leduc-deep-cfr/checkpoints/latest.pt \
    --iterations 250
```

### Modified-HULHE Deep CFR

Use the CPU smoke preset for a local lifecycle check:

```bash
ac-cfr-train \
    --config configs/deep_cfr/modified_hulhe_smoke.toml \
    --run-id modified-hulhe-smoke \
    --preflight

ac-cfr-train \
    --config configs/deep_cfr/modified_hulhe_smoke.toml \
    --run-id modified-hulhe-smoke
```

Preflight validates paths, resources, device execution and one small traversal without creating
the run. Production-scale modified-HULHE training requires deliberate GPU, storage and runtime
planning. The completed experiment is documented in the
[modified-HULHE results](../results/modified_hulhe/README.md).

## Plotting

Generate standard diagnostics for one run:

```bash
ac-cfr-plot-results runs/example-kuhn-cfr-plus
```

Compare exact exploitability against elapsed training time:

```bash
ac-cfr-plot-results \
    runs/leduc-cfr \
    runs/leduc-cfr-plus \
    --metric exploitability \
    --x-axis elapsed_training_seconds \
    --output runs/plots/leduc-exploitability.png
```

Result-specific READMEs under [`results/`](../results/) include the commands used to regenerate
committed figures from compact evidence.

## Benchmarks and validation suites

Run an individual tabular timing workload:

```bash
ac-cfr-benchmark \
    --game kuhn \
    --solver cfr_plus \
    --iterations 10000 \
    --repeats 5
```

Named suites reproduce the larger correctness, convergence, profiling and performance studies:

```bash
ac-cfr-benchmark \
    --suite cfr-cfr-plus \
    --output runs/benchmarks/cfr-cfr-plus
```

Some suites require substantial CPU, memory or GPU time. Supply an output directory under
`runs/` for exploratory execution so committed evidence is not replaced. Use
`ac-cfr-benchmark --help` for the available suites; their formal workloads and environments
are described under `results/`.

## File ownership

| Path | Contents | Version-controlled |
|---|---|---|
| `configs/` | Game, training and strategy-registry configuration | Yes |
| `results/` | Curated evidence, plots and result documentation | Yes |
| `runs/` | Local metrics, checkpoints, snapshots and experiment output | No |
| `artifacts/` | Downloaded or selected playable policies | No |

Generated policies and checkpoints stay outside Git. The deterministic Hold'em evaluator tables
under `src/ac_cfr/games/holdem/evaluator/data/` are committed package data.

## Development checks

```bash
ruff check .
ruff format --check .
pyright
pytest
```

The default test command excludes `slow`, `gpu`, `cloud` and `benchmark` tests. Run the slow
evaluator checks when changing the evaluator or its tables:

```bash
pytest -m slow
```

Build the wheel after packaging or entry-point changes:

```bash
python -m build --wheel
```

Regenerate the deterministic Hold'em lookup tables only when changing their generation or
evaluation logic:

```bash
python tools/generate_holdem_evaluator.py
pytest -m slow
```

CI runs formatting, linting, type checking, the default tests, an installed-wheel smoke test,
and a production-container build and health check.
