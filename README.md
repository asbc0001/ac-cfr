# AC CFR

A poker research project for implementing, validating, and comparing counterfactual regret minimisation algorithms. 

The current codebase provides the shared game, evaluation, information-state, randomness, and playable-agent foundations that later solvers will use.

## Current functionality

- Complete two-player Kuhn and Leduc engines with deterministic dense indexed trees.
- One configurable heads-up fixed-limit Hold'em engine supporting conventional HULHE and a reduced modified-HULHE game.
- A compact 52-card representation and fast seven-card Hold'em evaluator.
- Deterministic suit-canonical, player-visible Hold'em information states.
- A common playable-agent interface with a uniform-random baseline policy.

Solver, training, checkpoint, evaluation, and web-demo code has not been implemented yet.

## Modified HULHE

This reduced training game begins directly on the flop. Both players start with one small bet contributed to a two-small-bet pot, with no pre-flop betting history. Flop, turn, and river play use standard heads-up fixed-limit position and bet sizing, but each round permits only an opening bet and one raise.

The same engine also supports conventional HULHE from the pre-flop blinds with four betting levels per round. Full HULHE is available for rule validation and possible future experiments; modified HULHE is the planned Deep CFR training target.

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

The default test suite excludes expensive tests marked as slow. Run the full evaluator gate separately when changing evaluator code or lookup data:

```bash
pytest -m slow
```

CI also builds and installs the package outside the checkout to verify that the evaluator data is included and loads correctly.

## Architecture

```text
src/ac_cfr/
├── agents/          Frozen playable-policy interfaces and baseline agents
├── common/          Configuration labels and deterministic RNG handling
└── games/
    ├── base.py      Shared extensive-form game contracts
    ├── tree.py      Dense indexed-tree representation
    ├── kuhn.py      Canonical Kuhn rules and engine
    ├── leduc.py     Canonical physical-card Leduc rules and engine
    └── holdem/      Cards, engine, information states, canonicalisation, and evaluator
```

Game states contain complete underlying hand data. Agents and future solvers make decisions through separate `InformationState` values containing only information visible to the acting player. Kuhn and Leduc use precomputed indexed trees, while Hold'em uses compact on-demand state transitions instead of pre-enumerating its full game tree.

The fast Hold'em evaluator uses reproducible packaged lookup tables validated against independent reference implementations. Regenerate them with:

```bash
python tools/generate_holdem_evaluator.py
```

## Repository conventions

Development uses a single `main` branch. Run the local checks before each direct commit or push; CI verifies every push to `main`.

Compact results and figures may be version-controlled under `results/`. Downloaded strategies belong under ignored `artifacts/`, while checkpoints and other training output belong under ignored `runs/`. Small deterministic evaluator tables are committed, but generated model files are kept out of normal Git history and distributed separately.
