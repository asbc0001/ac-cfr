# Modified-HULHE exploratory opponent sampling

This experiment tested whether importance-corrected opponent exploration could reduce the betting-distribution mismatch between Deep CFR self-play and `rule_based_v1`. Rule-bot games contained more checked-to states, shallower pots and earlier-street decisions; hand-strength distributions differed much less.

Opponent actions were sampled from a mixture of the current policy (`sigma`) and 10% uniform exploration: `q = 0.9 sigma + 0.1 uniform`. Importance weights corrected sampled returns and training examples, while chance sampling, the updating player's action expansion, true-policy storage and uniform replacement in the bounded training memories remained unchanged. The estimator was validated independently on Leduc before modified-HULHE training.

Exploration improved results against random and some saved-policy opponents but did not materially close the passive/shallow distribution gap. Early rule-bot progression worsened before recovering later. At iteration 20, matched 8,000-update runs with and without exploration were effectively tied against the rule bot. Exploration performed better against random, while standard sampling produced more accurate held-out policy and advantage predictions.

The sampling implementation was therefore retained as an experimental option, but 10% exploration was not selected. The final policy used standard sampling without added exploration. Relevant policy comparisons are included in [`../final_policy/policy_evaluations.csv`](../final_policy/policy_evaluations.csv).

Presets: [`../../../configs/deep_cfr/modified_hulhe_exploratory_development.toml`](../../../configs/deep_cfr/modified_hulhe_exploratory_development.toml) and [`../../../configs/deep_cfr/modified_hulhe_exploratory_confirmation.toml`](../../../configs/deep_cfr/modified_hulhe_exploratory_confirmation.toml).
