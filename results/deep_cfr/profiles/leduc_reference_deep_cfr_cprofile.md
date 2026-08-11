# cProfile: reference Leduc Deep CFR

This automatically generated diagnostic records Python and native-call CPU time. It uses a separate warmed run and is not a formal runtime measurement.

- Outer iterations: 3
- Traversals: 3,000
- Optimizer steps: 220

```text
         19272801 function calls (18709861 primitive calls) in 31.678 seconds

   Ordered by: cumulative time
   List reduced from 352 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       13    0.014    0.001   52.318    4.024 naive_deep_cfr.py:721(_evaluate_network_loss)
       14    0.206    0.015   50.346    3.596 grad_mode.py:294(__exit__)
        7    1.352    0.193   24.213    3.459 naive_deep_cfr.py:654(_fit_network)
      220    0.002    0.000   13.343    0.061 _tensor.py:566(backward)
      220    0.007    0.000   13.341    0.061 __init__.py:255(backward)
      220    0.006    0.000   13.322    0.061 graph.py:966(_engine_run_backward)
      220   13.311    0.061   13.311    0.061 {method 'run_backward' of 'torch._C._EngineBase' objects}
272835/30315    0.166    0.000    6.949    0.000 module.py:1774(_wrapped_call_impl)
272835/30315    0.336    0.000    6.922    0.000 module.py:1782(_call_impl)
    30315    0.020    0.000    6.871    0.000 networks.py:95(forward)
    30315    0.123    0.000    6.784    0.000 container.py:248(forward)
64150/3000    0.317    0.000    6.306    0.002 naive_deep_cfr.py:294(_traverse)
   121260    0.080    0.000    5.671    0.000 linear.py:130(forward)
   121260    5.549    0.000    5.549    0.000 {built-in method torch._C._nn.linear}
      220    0.009    0.000    4.362    0.020 optimizer.py:509(wrapper)
      220    0.004    0.000    4.320    0.020 optimizer.py:60(_use_grad)
      220    0.004    0.000    4.314    0.020 adam.py:214(step)
      220    0.001    0.000    4.290    0.020 optimizer.py:131(maybe_fallback)
      220    0.003    0.000    4.289    0.019 adam.py:902(adam)
        1    0.000    0.000    4.288    4.288 naive_deep_cfr.py:574(_train_network_tensors)
      220    0.098    0.000    4.283    0.019 adam.py:347(_single_tensor_adam)
     1760    4.099    0.002    4.099    0.002 {method 'sqrt' of 'torch._C.TensorBase' objects}
    34284    0.148    0.000    2.911    0.000 naive_deep_cfr.py:266(strategy_for_information_set)
    34284    0.229    0.000    2.415    0.000 naive_deep_cfr.py:374(_predict_advantages)
    34284    0.076    0.000    2.059    0.000 reservoirs.py:417(_validate_common_sample)
   171860    0.255    0.000    2.018    0.000 {built-in method builtins.any}
    20207    0.027    0.000    1.430    0.000 reservoirs.py:408(__post_init__)
  1302792    0.272    0.000    1.428    0.000 reservoirs.py:420(<genexpr>)
3250282/3249837    0.406    0.000    1.312    0.000 {built-in method builtins.isinstance}
    14077    0.012    0.000    0.933    0.000 reservoirs.py:393(__post_init__)


```
