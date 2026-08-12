# Modified HULHE Cloud Calibration

This directory records the accepted hardware calibration and traversal-worker selection evidence for the optimised modified-HULHE Deep CFR pipeline. It does not contain a final strategy-training result.

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

The strategy-network budget remains provisionally 16,000 updates with batch size 10,000. The candidate is not frozen until the production shakedown, recovery test and policy-progression safeguards pass.

## Advantage-network fit

One K=10,000 collection produced 30,000 retained advantage samples for Player 0. A fixed 27,000/3,000 train/validation split trained one freshly initialised network continuously through all four milestones.

| Updates | Training loss | Validation loss | Cumulative time |
|---:|---:|---:|---:|
| 4,000 | 0.004590 | 2.144159 | 65.12 s |
| 8,000 | 0.022484 | 2.128387 | 127.99 s |
| 16,000 | 0.002959 | 2.080709 | 253.46 s |
| 32,000 | 0.002886 | **1.967588** | 504.96 s |

Validation loss improved by 5.4% from 16,000 to 32,000 updates, selecting the maximum declared 32,000-update budget. The large train/validation gap remains a shakedown diagnostic concern; low fitting loss alone is not evidence of strategy quality.

![Advantage-network fit](plots/network_fit.png)

## Generalisation investigation

The continuous 32,000-update fit was repeated across both players and two independently seeded collections. Every case retained 30,000 samples and used one disjoint fixed 27,000/3,000 split throughout its fit. Optimisation sampled only training indices, validation read only held-out indices under inference mode, and every sample had the same outer-iteration weight because collection occurred at iteration 1.

| Seed | Player | Best update | Training loss at 32k | Validation loss at 32k |
|---:|---:|---:|---:|---:|
| 20260811 | 0 | 32,000 | 0.002886 | 1.967588 |
| 20260811 | 1 | 16,000 | 0.002204 | 2.045822 |
| 20260812 | 0 | 32,000 | 0.002135 | 1.964542 |
| 20260812 | 1 | 32,000 | 0.002666 | 1.923552 |

The large training/validation gap is systematic rather than a single-player, single-reservoir or split-weighting anomaly. The 32,000-update milestone was best in three cases; Player 1 at seed 20260811 was marginally best at 16,000 and worsened by 0.35% at 32,000. The candidate therefore retains 32,000 updates provisionally. This evidence does not justify a larger model or lower learning rate, and K remains 10,000 until policy-level progression shows whether more diverse traversal data is necessary.

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

Three one-iteration runs kept K=10,000, model, seed, CUDA placement, inference batching, reservoirs and learning settings fixed while changing only the worker count. Neural fitting was bounded to 100 identical updates per phase because this check measures worker scaling rather than strategy quality.

| Workers | Traversal time | Collection throughput | Complete iteration |
|---:|---:|---:|---:|
| 8 | 33.00 s | 605.97 traversals/s | 76.89 s |
| 10 | 31.67 s | 631.45 traversals/s | 41.15 s |
| 12 | **29.53 s** | **677.33 traversals/s** | **39.12 s** |

Twelve workers improved collection throughput by 7.3% and complete-iteration time by 4.9% relative to ten workers. The eight-worker run incurred a one-time neural warm-up cost, so its separated traversal measurement is more representative than its complete-iteration time.

![Traversal-worker scaling](plots/worker_scaling.png)

## Hardware and boundaries

Calibration used Python 3.12.3, PyTorch 2.13.0, CUDA 13.0 and one NVIDIA A100-SXM4 with 80 GB VRAM. The container exposed a 13.6-core CPU quota, approximately 250 GB decimal RAM and a configured 200 GB storage budget. Full environment and resolved configuration data are preserved in `calibration.json`.

These results select a bounded production candidate; they do not show that modified-HULHE policy quality improves monotonically or approaches Nash equilibrium. The production shakedown and modified-HULHE duplicate-deal progression measurements remain mandatory before the long run.

## Progression evaluation

The evaluation command now accepts validated modified-HULHE average-strategy snapshots and can compare them against the fixed uniform-random baseline, explicit fixed snapshot anchors, themselves for neutrality, and selected earlier snapshots in an ordered round-robin. Each complete physical deal is replayed with the agents' seats and button assignments swapped while retaining the same seat-labelled private cards and public runout. The two focal-policy outcomes form one independent duplicate-pair score.

Results report `mbb/g` with a predeclared seeded paired-bootstrap confidence interval and atomically upsert into a compact CSV. Plot generation reads that CSV directly. No rule-based opponent is included yet because the repository does not contain a defensible fixed rule-based Hold'em agent; adding one remains conditional on it being proportionate.

Example after shakedown snapshots exist:

~~~bash
python evaluate.py modified-hulhe \
  --snapshot runs/<run-id>/strategy_snapshots/<early>.pt \
  --snapshot runs/<run-id>/strategy_snapshots/<later>.pt \
  --include-random \
  --include-self-play \
  --duplicate-pairs 10000 \
  --seed 20260811 \
  --confidence-level 0.95 \
  --bootstrap-resamples 10000 \
  --results runs/<run-id>/evaluation/h2h.csv

python plot_results.py modified-hulhe-h2h \
  runs/<run-id>/evaluation/h2h.csv \
  --output runs/<run-id>/evaluation/h2h.png
~~~

Use `--anchor-snapshot PATH` for a fixed earlier policy. Supplying multiple `--snapshot` values additionally evaluates each later iteration against every selected earlier iteration. The shakedown will choose the formal pair count, confidence level, bootstrap count, seeds and milestones before compact accepted measurements are copied into this results directory.

## Files

- `calibration.json` records the source revision, environment, resolved configuration, collection measurements and calibration method.
- `network_fit.csv` contains the original full-precision continuous-fit milestones.
- `generalisation_fit.csv` contains all four full-precision seed/player fit curves.
- `generalisation_fit.json` records the split, weighting, reservoir provenance and resulting decision.
- `batch_throughput.csv` contains the full-precision warmed GPU batch measurements.
- `worker_scaling.json` records the controlled comparison method and selection.
- `worker_scaling.csv` contains the full-precision worker measurements.
- `plots/` contains compact visual summaries generated from the corresponding CSV files.

Raw checkpoints, snapshots, logs and run-local metrics remain under ignored `runs/` directories.
