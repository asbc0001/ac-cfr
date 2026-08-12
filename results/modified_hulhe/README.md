# Modified HULHE

This directory records calibration and policy-evaluation evidence for the optimised modified-HULHE Deep CFR pipeline. It does not yet contain a final training result.

## Status

- **Advantage-update calibration:** COMPLETE
- **GPU batch calibration:** COMPLETE
- **Traversal-worker selection:** COMPLETE
- **Generalisation investigation:** COMPLETE
- **Policy-progression evaluator:** COMPLETE
- **Production shakedown and policy measurements:** PENDING
- **Long cloud run:** NOT STARTED

## Selected candidate

| Setting | Selection | Evidence |
|---|---:|---|
| Traversals per player (`K`) | 10,000 | Representative collection workload |
| Advantage SGD steps | 32,000 | Held-out loss continued improving from 16,000 to 32,000 updates |
| Advantage batch size | 10,000 | 20,000 was materially slower |
| Learning rate | 0.001 | Finite and stable calibration |
| Gradient clip norm | 1.0 | Finite and stable calibration |
| Traversal workers | 12 | Highest measured traversal and complete-iteration throughput |

The provisional strategy-network budget is 16,000 updates with batch size 10,000. The configuration will be frozen only after the production shakedown, recovery test and policy-progression checks pass.

## Advantage-network fit

One K=10,000 collection produced 30,000 retained advantage samples for Player 0. A fixed 27,000/3,000 train/validation split trained one freshly initialised network continuously through all four milestones.

| Updates | Training loss | Validation loss | Cumulative time |
|---:|---:|---:|---:|
| 4,000 | 0.004590 | 2.144159 | 65.12 s |
| 8,000 | 0.022484 | 2.128387 | 127.99 s |
| 16,000 | 0.002959 | 2.080709 | 253.46 s |
| 32,000 | 0.002886 | **1.967588** | 504.96 s |

Validation loss improved by 5.4% from 16,000 to 32,000 updates, selecting the declared maximum of 32,000. The large train/validation gap remains a shakedown concern; low training loss alone does not establish strategy quality.

![Advantage-network fit](plots/network_fit.png)

## Generalisation investigation

The continuous 32,000-update fit was repeated for both players across two independently seeded collections. Each case retained 30,000 samples and used a fixed, disjoint 27,000/3,000 training/validation split. All samples came from iteration 1 and therefore had equal Linear CFR weight.

| Seed | Player | Best update | Training loss at 32k | Validation loss at 32k |
|---:|---:|---:|---:|---:|
| 20260811 | 0 | 32,000 | 0.002886 | 1.967588 |
| 20260811 | 1 | 16,000 | 0.002204 | 2.045822 |
| 20260812 | 0 | 32,000 | 0.002135 | 1.964542 |
| 20260812 | 1 | 32,000 | 0.002666 | 1.923552 |

The large training/validation gap is systematic rather than isolated to one player, reservoir or split. The 32,000-update milestone was best in three cases; Player 1 at seed 20260811 was marginally best at 16,000 and worsened by 0.35% at 32,000. The candidate therefore retains 32,000 updates provisionally. There is no evidence yet for changing the model or learning rate, while K remains 10,000 pending policy-level progression results.

![Advantage-network generalisation](plots/generalisation_fit.png)

## GPU batch selection

Both batch sizes used identically initialised networks, the same frozen reservoir and split seed, 50 warm-up steps and 200 timed updates.

| Batch | Updates/s | Median GPU utilisation | Peak allocated VRAM | Peak reserved VRAM |
|---:|---:|---:|---:|---:|
| 10,000 | **63.91** | 52% | 256.0 MB | 421.5 MB |
| 20,000 | 39.18 | 58% | 472.6 MB | 639.6 MB |

Batch 20,000 was 38.7% slower despite slightly higher GPU utilisation, so the candidate retains batch 10,000.

![GPU batch throughput](plots/batch_throughput.png)

## Traversal-worker selection

Three one-iteration runs changed only the worker count. Neural fitting was limited to 100 identical updates per phase because this check measures worker scaling, not strategy quality.

| Workers | Traversal time | Collection throughput | Complete iteration |
|---:|---:|---:|---:|
| 8 | 33.00 s | 605.97 traversals/s | 76.89 s |
| 10 | 31.67 s | 631.45 traversals/s | 41.15 s |
| 12 | **29.53 s** | **677.33 traversals/s** | **39.12 s** |

Twelve workers improved collection throughput by 7.3% and complete-iteration time by 4.9% relative to ten workers. The eight-worker run incurred a one-time neural warm-up cost, so its separated traversal measurement is more representative than its complete-iteration time.

![Traversal-worker scaling](plots/worker_scaling.png)

## Hardware and boundaries

Calibration used Python 3.12.3, PyTorch 2.13.0, CUDA 13.0 and one NVIDIA A100-SXM4 with 80 GB VRAM. The container provided a 13.6-core CPU quota, approximately 250 GB decimal RAM and a configured 200 GB storage budget. `calibration.json` preserves the full environment and resolved configuration.

These results select a bounded production candidate; they do not show that modified-HULHE policy quality improves monotonically or approaches Nash equilibrium. The production shakedown and modified-HULHE duplicate-deal progression measurements remain mandatory before the long run.

## Shakedown monitoring

The three-iteration shakedown uses `configs/deep_cfr/modified_hulhe_shakedown.toml`. Training must stop on non-finite values, invalid actions or state, worker or checkpoint failure, or exhausted memory/storage. These conditions already raise from the solver, game, worker, preflight or checkpoint boundary; they are not converted into warnings.

Policy measurements are diagnostic rather than automatic stopping conditions. One or two disappointing H2H estimates must be recorded and investigated, not used to stop training. Only sustained stagnation or instability across several snapshots can justify one K=30,000 diagnostic. Any response changes one relevant configuration value while holding the others fixed; model size, learning rate and regularisation remain unchanged without corresponding evidence.

Each completed iteration records training and held-out losses, phase times, traversal throughput, reservoir growth, process memory, CUDA peak memory and remaining storage. Checkpoints and average-strategy snapshots are written after every shakedown iteration.

## Progression evaluation

The evaluator compares validated average-strategy snapshots against uniform random, fixed snapshot anchors, themselves for neutrality and selected earlier snapshots. Each physical deal is replayed with the agents swapped between seats while retaining the same seat-labelled private cards and public runout. The two focal-policy outcomes form one independent duplicate-pair score.

Results report `mbb/g` with seeded paired-bootstrap confidence intervals and are atomically upserted into a compact CSV used directly for plotting. A rule-based opponent remains deferred because no defensible fixed implementation currently exists.

The fixed shakedown protocol uses 10,000 duplicate pairs, seed 20260811, a 95% interval and 10,000 bootstrap resamples. Multiple snapshots evaluate each later policy against every selected earlier policy; `--anchor-snapshot PATH` can add a fixed policy from another run.

## Cloud shakedown commands

Run these commands from the checked-out repository at `/workspace/ac-cfr`. Preflight validates the resolved production workload without creating a run:

```bash
.venv/bin/python train.py \
  --config configs/deep_cfr/modified_hulhe_shakedown.toml \
  --run-id modified-hulhe-shakedown \
  --runs-root /workspace/ac-cfr/runs \
  --preflight
```

Start the shakedown in a detached persistent session:

```bash
tmux new-session -d -s hulhe-shakedown \
  "cd /workspace/ac-cfr && exec .venv/bin/python train.py \
  --config configs/deep_cfr/modified_hulhe_shakedown.toml \
  --run-id modified-hulhe-shakedown \
  --runs-root /workspace/ac-cfr/runs"

tmux attach-session -t hulhe-shakedown
```

For the interruption test, wait until at least one iteration has completed, then identify the coordinator process attached to the tmux pane:

```bash
tmux display-message -p -t hulhe-shakedown '#{pane_pid}'
ps -o pid,ppid,pgid,stat,etime,cmd -p <PID>
kill -TERM <PID>
```

Confirm from the `ps` output that `<PID>` is the training coordinator before signalling it. Sending `SIGTERM` only to that process avoids forwarding terminal `SIGINT` to traversal workers. The coordinator requests a stop at the end of the active outer iteration, writes its snapshot, checkpoint and metrics, then exits. Resume the immutable saved configuration after it has stopped:

```bash
tmux new-session -d -s hulhe-shakedown-resume \
  "cd /workspace/ac-cfr && exec .venv/bin/python train.py \
  --resume /workspace/ac-cfr/runs/modified-hulhe-shakedown/checkpoints/latest.pt"
```

After all three snapshots exist, run the seeded duplicate-deal progression evaluation and plot its results:

```bash
.venv/bin/python evaluate.py modified-hulhe \
  --snapshot runs/modified-hulhe-shakedown/strategy_snapshots/modified-hulhe-shakedown_iter_1.pt \
  --snapshot runs/modified-hulhe-shakedown/strategy_snapshots/modified-hulhe-shakedown_iter_2.pt \
  --snapshot runs/modified-hulhe-shakedown/strategy_snapshots/modified-hulhe-shakedown_iter_3.pt \
  --include-random \
  --include-self-play \
  --duplicate-pairs 10000 \
  --seed 20260811 \
  --confidence-level 0.95 \
  --bootstrap-resamples 10000 \
  --results runs/modified-hulhe-shakedown/evaluation/h2h.csv

.venv/bin/python plot_results.py modified-hulhe-h2h \
  runs/modified-hulhe-shakedown/evaluation/h2h.csv \
  --output runs/modified-hulhe-shakedown/evaluation/h2h.png
```

## Files

- `calibration.json`: source revision, environment, resolved configuration and calibration method.
- `network_fit.csv`: original continuous-fit milestones.
- `generalisation_fit.csv` and `generalisation_fit.json`: seed/player fit curves, split and reservoir provenance, and resulting decision.
- `batch_throughput.csv`: warmed GPU batch measurements.
- `worker_scaling.csv` and `worker_scaling.json`: controlled worker comparison and selection.
- `plots/`: visual summaries generated from the CSV files.

Raw checkpoints, snapshots, logs and run-local metrics remain under ignored `runs/` directories.
