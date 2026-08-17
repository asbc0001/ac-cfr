# Modified-HULHE Deep CFR

This directory contains compact training and evaluation evidence for the modified heads-up limit Hold'em Deep CFR pipeline.

## Directory layout

- [`calibration/`](calibration/) contains the network-fit, batch-size and traversal-worker experiments used to choose the initial training settings.
- [`operational_test/`](operational_test/) contains a three-iteration test of training, interruption recovery and policy evaluation, together with its configuration, environment and metrics.
- [`final_policy/`](final_policy/) contains the completed 240-iteration run summary, per-iteration and selected-iteration metrics, policy comparisons, plots and artifact manifest.
- [`exploratory_sampling/`](exploratory_sampling/) summarises the completed importance-corrected sampling experiment, which was not selected for the final run.

Raw run data is kept under ignored `runs/` directories. The selected playable snapshot belongs under ignored `artifacts/` and is verified through the strategy registry.

## Outcome

The [final run](final_policy/run_summary.json) completed 240 outer iterations and 4.8 million traversals. Iteration 240 was selected because the late snapshots showed no sustained decline that justified using an earlier policy.

Against the fixed `rule_based_v1` baseline, iteration 240 achieved 151.70 mbb/g with a 95% confidence interval of [126.00, 177.70]. It achieved 562.15 mbb/g against uniform random. These are opponent-specific playing-strength measurements, not exploitability estimates or evidence of proximity to Nash equilibrium.

## Final configuration

| Setting | Value |
|---|---:|
| Traversals per player per iteration (`K`) | 10,000 |
| Advantage updates per player | 8,000 |
| Advantage batch size | 10,000 |
| Strategy updates at snapshot boundaries | 4,000 |
| Learning rate | 0.001 |
| Gradient clipping | 1.0 |
| Opponent exploration probability | 0.0 |
| All-negative fallback mixing | 0.0 |
| Dropout | 0.0 |
| Traversal workers | 12 |
| Device | NVIDIA A100-SXM4-80GB |

Information-state-grouped validation kept every occurrence of a canonical information state entirely in either training or validation, testing the networks on unseen states rather than repeated examples of training states.

## Policy progression

`mbb/g` is thousandths of one small bet won per game. Each physical deal was played twice with the agents swapping seats. Every value below uses 10,000 duplicate pairs, seed 20260811 and a 95% paired-bootstrap confidence interval from 10,000 resamples.

| Iteration | vs rule bot (mbb/g) | 95% confidence interval | vs random (mbb/g) |
|---:|---:|---:|---:|
| 1 | -114.40 | [-145.90, -82.40] | — |
| 2 | -104.60 | [-141.40, -67.75] | — |
| 5 | -224.70 | [-260.30, -189.25] | — |
| 20 | -116.60 | [-148.65, -85.25] | 555.65 |
| 40 | 38.90 | [8.75, 69.55] | — |
| 60 | 70.45 | [41.50, 99.20] | — |
| 80 | 113.20 | [85.95, 140.15] | — |
| 100 | 135.60 | [107.80, 163.60] | — |
| 130 | 130.15 | [104.05, 155.95] | 566.75 |
| 150 | 153.70 | [127.65, 179.20] | 593.60 |
| 180 | 148.65 | [123.55, 173.90] | 549.70 |
| 210 | 133.65 | [107.75, 159.60] | 605.15 |
| 240 | **151.70** | **[126.00, 177.70]** | 562.15 |

The complete [policy evaluations](final_policy/policy_evaluations.csv) and [same-deal comparisons](final_policy/paired_differences.csv) found a decline from iteration 180 to 210 of -15.00 mbb/g [-27.40, -2.65], followed by a recovery from 210 to 240 of 18.05 mbb/g [6.45, 29.70]. Direct snapshot matches placed iteration 240 on the strongest observed plateau: it beat iteration 100 by 51.55 mbb/g [19.30, 83.45] and was statistically tied with iterations 150, 180 and 210. Its lower random-agent score than iteration 210 was opponent-specific and did not justify replacing the final snapshot.

![Policy progression](final_policy/policy_progression.png)

## Training health

- Final Player 0 advantage loss: 10.593 training and 11.950 grouped validation.
- Final Player 1 advantage loss: 22.518 training and 23.699 grouped validation.
- Final strategy loss: 0.2266 training and 0.2364 validation.
- The bounded strategy, Player 1 advantage and Player 0 advantage training memories reached their 10-million-sample capacities at iterations 36, 73 and 169.
- Median iteration time was 370.05 seconds, including 15.85 seconds of traversal and 318.25 seconds of advantage fitting.
- Recorded solver training time was 23.22 hours. Network-volume and checkpoint stalls increased recorded iteration wall time to 30.61 hours.
- Peak process memory was 13.44 GiB. Peak CUDA allocation and reservation were 295.5 MiB and 614 MiB.
- All recorded resource, loss and timing values were finite.

Absolute advantage losses increased late because the regret targets and their scale evolved during training. Policy evaluation is therefore the primary evidence of playing strength.

## Calibration and investigation

[Calibration](calibration/) selected the 10,000-sample advantage batch and 12 traversal workers on the A100 training environment. A 20,000-sample batch was 38.7% slower, while 12 workers delivered 677.33 traversals per second and the fastest complete calibration iteration.

The [three-iteration operational test](operational_test/) verified finite training, loadable snapshots and checkpoint recovery after a deliberate stop and resume. It improved against random and earlier snapshots while remaining weaker than the rule bot.

The early rule-bot decline remained significant when identical duplicate deals were replayed, prompting targeted follow-up experiments:

| Investigation | Result | Decision |
|---|---|---|
| Encoder, state coverage and target estimator | No meaningful collision, state-transition or estimator-bias bug was found; individual advantage targets were very noisy | Keep the engine, encoding and traversal semantics |
| 20,000 traversals per player | Improved some neural and policy measurements but did not fix the rule-bot trend and cost about 13% more runtime | Retain 10,000 traversals per player |
| Structured or smaller networks, dropout, final-layer LayerNorm and mixed fallback | Isolated offline gains did not translate into better rule-bot progression | Retain the flat network, no dropout and deterministic fallback |
| Training the playable average-policy network | Longer fitting improved rule-bot H2H but worsened held-out probability-distribution accuracy; policies played directly from advantage networks were stronger but still deteriorated later | Select the fitting duration using held-out accuracy rather than one opponent |
| [Exploratory opponent sampling](exploratory_sampling/) | The corrected estimator was valid, but 10% exploration did not close the passive/shallow state-distribution gap or improve rule-bot results over matched standard sampling | Retain standard sampling without added exploration |
| Advantage fitting and training horizon | Grouped validation favoured 8,000 over 32,000 updates; fitting was about 3.8 times faster and enabled about 3.6 times more iterations per hour | Use 8,000 updates and evaluate over a longer horizon |

The main issue was therefore not a single engine or sampling defect. Modified-HULHE produced noisy advantage targets, and the playable average-policy network did not reproduce its training data perfectly, making short runs misleading. The standard 8,000-update run was still about -118 mbb/g against the rule bot at iteration 20, but reached 135.6 mbb/g [107.8, 163.6] by iteration 100 and remained strong through iteration 240.

## Integrity and limitations

The final 12 GB checkpoint was loaded and reconstructed during final analysis before the compact evidence was exported. It is not included in this results directory. All three bounded training memories contained 10 million samples, the final strategy network was present, and all 25 scheduled strategy snapshots existed. The iteration-240 snapshot was also loaded for H2H evaluation.

The external iteration metrics contain 239 rows because iteration 228 is absent. Checkpoint publication failed after the internal metric and solver state were saved but before the external timing row was appended. Training resumed from the validated iteration-228 checkpoint, preserving policy-state continuity.

The run recorded source revision `ace54fa141fb377914aa590ea0358813b0e621b6-dirty`. The artifact manifest preserves the original run-relative paths, hashes and provenance; its paths describe the source archive rather than this compact directory layout. All presentation plots were regenerated from the preserved CSV evidence through `plot_results.py`.

The rule-based agent is deterministic and uses only visible cards and public betting state; it does not use hidden cards, learned-policy information, opponent modelling or Monte Carlo equity. Results against it measure performance against one fixed policy. Modified-HULHE H2H does not measure exploitability, and the reported matches use one evaluation seed.
