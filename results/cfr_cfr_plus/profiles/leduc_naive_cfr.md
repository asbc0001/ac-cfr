# CPU profile: Leduc naive_cfr

This file is generated automatically with Python's `cProfile`. It shows where the solver spent CPU time during a separate diagnostic run and is not part of the formal benchmark timing.

- **Solver:** `naive_cfr`
- **Game:** Leduc
- **Iterations:** 100
- **Rows:** top 25 functions, sorted by cumulative time

`ncalls` is the number of calls. `tottime` is time spent inside the function itself. `cumtime` includes functions it called. Each `percall` column divides the adjacent time by its relevant call count. Times are in seconds.

## Raw cProfile output

```text
         14000597 function calls (12110597 primitive calls) in 5.301 seconds

   Ordered by: cumulative time
   List reduced from 26 to 25 due to restriction <25>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.017    0.017    5.301    5.301 naive_cfr.py:44(train)
      200    0.001    0.000    5.283    0.026 naive_cfr.py:85(_run_player_pass)
1890200/200    2.979    0.000    4.612    0.023 naive_cfr.py:107(_traverse)
  1890200    0.396    0.000    0.678    0.000 enum.py:726(__call__)
   943200    0.334    0.000    0.552    0.000 {built-in method math.fsum}
      200    0.015    0.000    0.461    0.002 naive_cfr.py:234(_regret_matched_policy)
   187400    0.126    0.000    0.445    0.000 naive_cfr.py:235(<genexpr>)
  1890200    0.282    0.000    0.282    0.000 enum.py:1129(__new__)
   378000    0.147    0.000    0.235    0.000 naive_cfr.py:197(_record_average_strategy)
  2520000    0.219    0.000    0.219    0.000 naive_cfr.py:184(<genexpr>)
   187200    0.118    0.000    0.192    0.000 naive_cfr.py:283(_normalise)
  1764000    0.189    0.000    0.189    0.000 {method 'append' of 'list' objects}
   624000    0.075    0.000    0.127    0.000 naive_cfr.py:236(<genexpr>)
      400    0.107    0.000    0.107    0.000 naive_cfr.py:240(_empty_table)
   378000    0.057    0.000    0.057    0.000 {method 'get' of 'dict' objects}
      200    0.054    0.000    0.054    0.000 naive_cfr.py:218(_apply_regret_delta)
   436800    0.053    0.000    0.053    0.000 {built-in method builtins.max}
   618424    0.052    0.000    0.052    0.000 naive_cfr.py:286(<genexpr>)
      200    0.048    0.000    0.048    0.000 naive_cfr.py:223(_apply_strategy_delta)
   284400    0.031    0.000    0.031    0.000 {built-in method math.isclose}
     5576    0.001    0.000    0.001    0.000 naive_cfr.py:288(<genexpr>)
     1692    0.000    0.000    0.000    0.000 {built-in method builtins.len}
      100    0.000    0.000    0.000    0.000 naive_cfr.py:228(_averaging_weight)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.000    0.000    0.000    0.000 naive_cfr.py:291(_validate_non_negative_integer)


```
