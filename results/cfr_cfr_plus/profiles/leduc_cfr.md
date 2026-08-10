# CPU profile: Leduc cfr

This file is generated automatically with Python's `cProfile`. It shows where the solver spent CPU time during a separate diagnostic run and is not part of the formal benchmark timing.

- **Solver:** `cfr`
- **Game:** Leduc
- **Iterations:** 100
- **Rows:** top 25 functions, sorted by cumulative time

`ncalls` is the number of calls. `tottime` is time spent inside the function itself. `cumtime` includes functions it called. Each `percall` column divides the adjacent time by its relevant call count. Times are in seconds.

## Raw cProfile output

```text
         305 function calls in 0.017 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.017    0.017 cfr.py:57(train)
      200    0.017    0.000    0.017    0.000 cfr.py:104(_run_player_pass)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
      100    0.000    0.000    0.000    0.000 cfr.py:141(_averaging_weight)
        1    0.000    0.000    0.000    0.000 cfr.py:321(_validate_non_negative_integer)
        2    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}


```
