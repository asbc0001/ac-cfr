# CPU profile: Leduc naive_cfr_plus

This file is generated automatically with Python's `cProfile`. It shows where the solver spent CPU time during a separate diagnostic run and is not part of the formal benchmark timing.

- **Solver:** `naive_cfr_plus`
- **Game:** Leduc
- **Iterations:** 100
- **Rows:** top 25 functions, sorted by cumulative time

`ncalls` is the number of calls. `tottime` is time spent inside the function itself. `cumtime` includes functions it called. Each `percall` column divides the adjacent time by its relevant call count. Times are in seconds.

## Raw cProfile output

```text
         14437293 function calls (12547293 primitive calls) in 5.560 seconds

   Ordered by: cumulative time
   List reduced from 26 to 25 due to restriction <25>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.018    0.018    5.560    5.560 naive_cfr.py:44(train)
      200    0.002    0.000    5.542    0.028 naive_cfr.py:85(_run_player_pass)
1890200/200    3.104    0.000    4.767    0.024 naive_cfr.py:107(_traverse)
  1890200    0.396    0.000    0.697    0.000 enum.py:726(__call__)
   943200    0.346    0.000    0.565    0.000 {built-in method math.fsum}
      200    0.016    0.000    0.497    0.002 naive_cfr.py:234(_regret_matched_policy)
   187400    0.130    0.000    0.481    0.000 naive_cfr.py:235(<genexpr>)
  1890200    0.301    0.000    0.301    0.000 enum.py:1129(__new__)
   378000    0.148    0.000    0.239    0.000 naive_cfr.py:197(_record_average_strategy)
   187200    0.129    0.000    0.228    0.000 naive_cfr.py:283(_normalise)
  2520000    0.219    0.000    0.219    0.000 naive_cfr.py:184(<genexpr>)
  1764000    0.184    0.000    0.184    0.000 {method 'append' of 'list' objects}
      200    0.089    0.000    0.135    0.001 naive_cfr_plus.py:20(_apply_regret_delta)
   624000    0.072    0.000    0.122    0.000 naive_cfr.py:236(<genexpr>)
   873700    0.096    0.000    0.096    0.000 {built-in method builtins.max}
      400    0.092    0.000    0.092    0.000 naive_cfr.py:240(_empty_table)
   619080    0.076    0.000    0.076    0.000 naive_cfr.py:286(<genexpr>)
   378000    0.059    0.000    0.059    0.000 {method 'get' of 'dict' objects}
      200    0.050    0.000    0.050    0.000 naive_cfr.py:223(_apply_strategy_delta)
   284400    0.032    0.000    0.032    0.000 {built-in method math.isclose}
     4920    0.001    0.000    0.001    0.000 naive_cfr.py:288(<genexpr>)
      100    0.000    0.000    0.000    0.000 naive_cfr_plus.py:33(_averaging_weight)
     1488    0.000    0.000    0.000    0.000 {built-in method builtins.len}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.000    0.000    0.000    0.000 naive_cfr.py:291(_validate_non_negative_integer)


```
