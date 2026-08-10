# CPU profile: Leduc mccfr

This file is generated automatically with Python's `cProfile`. It records a separate diagnostic run and does not affect the formal benchmark timing.

- **Solver:** `mccfr`
- **Game:** Leduc
- **Iterations:** 100,000
- **Traversals:** 200,000
- **Rows:** top 25 functions, sorted by cumulative time

`ncalls` is the call count. `tottime` is time inside a function; `cumtime` also includes functions it called. Times are seconds. Numba-compiled operations are not individually visible to `cProfile`.

## Raw cProfile output

```text
         85 function calls in 0.228 seconds

   Ordered by: cumulative time
   List reduced from 26 to 25 due to restriction <25>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.228    0.228    0.228    0.228 mccfr.py:70(train)
        2    0.000    0.000    0.000    0.000 dispatcher.py:678(typeof_pyval)
        2    0.000    0.000    0.000    0.000 typeof.py:27(typeof)
        2    0.000    0.000    0.000    0.000 functools.py:904(wrapper)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        2    0.000    0.000    0.000    0.000 typeof.py:290(typeof_random_generator)
        2    0.000    0.000    0.000    0.000 abstract.py:61(__call__)
       12    0.000    0.000    0.000    0.000 __init__.py:517(cast)
        2    0.000    0.000    0.000    0.000 abstract.py:49(_intern)
        2    0.000    0.000    0.000    0.000 {method 'get' of 'dict' objects}
        2    0.000    0.000    0.000    0.000 functools.py:818(dispatch)
        6    0.000    0.000    0.000    0.000 __init__.py:73(CFUNCTYPE)
        4    0.000    0.000    0.000    0.000 abstract.py:121(__hash__)
        2    0.000    0.000    0.000    0.000 npytypes.py:635(__init__)
        1    0.000    0.000    0.000    0.000 cfr.py:327(_validate_non_negative_integer)
        2    0.000    0.000    0.000    0.000 {method 'add' of 'set' objects}
        2    0.000    0.000    0.000    0.000 abstract.py:124(__eq__)
        2    0.000    0.000    0.000    0.000 weakref.py:414(__getitem__)
       12    0.000    0.000    0.000    0.000 {method 'pop' of 'dict' objects}
        2    0.000    0.000    0.000    0.000 <string>:1(<lambda>)
        4    0.000    0.000    0.000    0.000 {built-in method __new__ of type object at 0xa44b40}
        8    0.000    0.000    0.000    0.000 abstract.py:96(key)
        4    0.000    0.000    0.000    0.000 {built-in method builtins.hash}
        2    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 {built-in method _abc.get_cache_token}


```
