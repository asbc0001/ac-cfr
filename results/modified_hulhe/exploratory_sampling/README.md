# Modified-HULHE exploratory sampling check

This short experiment checks whether importance-corrected opponent exploration reduces the passive and shallow-state mismatch observed in modified HULHE. It does not train against the rule-based agent or change the failed Leduc efficacy result.

The development and confirmation presets each run two iterations with `K = 10,000` and `epsilon = 0.1`, using seeds `20260811` and `20260812`. Other training settings match the shakedown run. Training records importance ratios, effective sample size, validation losses, and separate raw and weighted advantage/strategy counts by street, whether facing a wager, pot size and betting level.

Review iteration 1 to 2 for:

- whether weighted coverage moves toward the passive, smaller-pot and lower-betting-level states observed during matched rule-bot evaluation;
- importance-ratio percentiles, maximum weight and effective sample size;
- paired H2H against the matching `epsilon = 0` snapshot on the same duplicate deals;
- iteration 1 to 2 performance against the rule bot, with random-agent and matched-snapshot games as learning sanity checks.

Matched H2H uses 10,000 duplicate pairs, seed `20260811`, 95% paired-bootstrap intervals and 10,000 bootstrap resamples. Baseline evidence and configuration are in [`../shakedown_h2h.csv`](../shakedown_h2h.csv) and [`../shakedown_run_config.json`](../shakedown_run_config.json).

There is no automatic pass threshold. Continue to iteration 5 only if coverage moves in the intended direction without clear rule-bot deterioration or signs of unhealthy learning.

Presets: [`../../../configs/deep_cfr/modified_hulhe_exploratory_development.toml`](../../../configs/deep_cfr/modified_hulhe_exploratory_development.toml) and [`../../../configs/deep_cfr/modified_hulhe_exploratory_confirmation.toml`](../../../configs/deep_cfr/modified_hulhe_exploratory_confirmation.toml).
