# CPU profile: Leduc naive_mccfr

This file is generated automatically with Python's `cProfile`. It records a separate diagnostic run and does not affect the formal benchmark timing.

- **Solver:** `naive_mccfr`
- **Game:** Leduc
- **Iterations:** 100,000
- **Traversals:** 200,000
- **Rows:** top 25 functions, sorted by cumulative time

`ncalls` is the call count. `tottime` is time inside a function; `cumtime` also includes functions it called. Times are seconds. Numba-compiled operations are not individually visible to `cProfile`.

## Raw cProfile output

```text
         40902598 function calls (36829168 primitive calls) in 23.965 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.103    0.103   23.965   23.965 naive_mccfr.py:53(train)
   200000    0.091    0.000   23.862    0.000 naive_mccfr.py:102(_run_player_traversal)
4273430/200000   11.857    0.000   23.771    0.000 naive_mccfr.py:111(_traverse)
  2013796    3.283    0.000    3.684    0.000 naive_mccfr.py:227(_sample_position)
  2298578    2.265    0.000    3.552    0.000 naive_cfr.py:291(_normalise)
  4273430    1.327    0.000    2.279    0.000 enum.py:726(__call__)
  3199269    1.063    0.000    1.478    0.000 {built-in method math.fsum}
  5379134    1.109    0.000    1.109    0.000 {built-in method builtins.max}
  4273430    0.952    0.000    0.952    0.000 enum.py:1129(__new__)
  7667898    0.837    0.000    0.837    0.000 naive_cfr.py:294(<genexpr>)
  2960325    0.415    0.000    0.415    0.000 naive_mccfr.py:160(<genexpr>)
  1397887    0.261    0.000    0.261    0.000 {method 'get' of 'dict' objects}
  2016744    0.231    0.000    0.231    0.000 {built-in method builtins.len}
   938858    0.170    0.000    0.170    0.000 {method 'random' of '_random.Random' objects}
     9814    0.001    0.000    0.001    0.000 naive_cfr.py:296(<genexpr>)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        1    0.000    0.000    0.000    0.000 naive_cfr.py:299(_validate_non_negative_integer)
        2    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}


```
