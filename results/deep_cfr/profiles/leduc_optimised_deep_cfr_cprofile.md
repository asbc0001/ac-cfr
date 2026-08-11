# cProfile: optimised Leduc Deep CFR

This automatically generated diagnostic records Python and native-call CPU time. It uses a separate warmed run and is not a formal runtime measurement.

- Outer iterations: 3
- Traversals: 3,000
- Optimizer steps: 220

```text
         1513214 function calls (1277085 primitive calls) in 30.051 seconds

   Ordered by: cumulative time
   List reduced from 354 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        7    1.395    0.199   25.584    3.655 naive_deep_cfr.py:654(_fit_network)
      220    0.002    0.000   14.140    0.064 _tensor.py:566(backward)
      220    0.009    0.000   14.138    0.064 __init__.py:255(backward)
      220    0.006    0.000   14.117    0.064 graph.py:966(_engine_run_backward)
      220   14.106    0.064   14.106    0.064 {method 'run_backward' of 'torch._C._EngineBase' objects}
      8/3    0.001    0.000   12.303    4.101 naive_deep_cfr.py:721(_evaluate_network_loss)
     19/3    0.188    0.010    9.674    3.225 grad_mode.py:294(__exit__)
 3285/365    0.016    0.000    8.241    0.023 module.py:1774(_wrapped_call_impl)
 3285/365    0.029    0.000    8.239    0.023 module.py:1782(_call_impl)
      365    0.002    0.000    8.232    0.023 networks.py:95(forward)
      365    0.024    0.000    8.227    0.023 container.py:248(forward)
     1460    0.006    0.000    7.852    0.005 linear.py:130(forward)
     1460    7.843    0.005    7.843    0.005 {built-in method torch._C._nn.linear}
        1    0.000    0.000    4.921    4.921 naive_deep_cfr.py:574(_train_network_tensors)
      220    0.012    0.000    4.608    0.021 optimizer.py:509(wrapper)
      220    0.006    0.000    4.552    0.021 optimizer.py:60(_use_grad)
      220    0.003    0.000    4.544    0.021 adam.py:214(step)
      220    0.001    0.000    4.518    0.021 optimizer.py:131(maybe_fallback)
      220    0.003    0.000    4.517    0.021 adam.py:902(adam)
      220    0.132    0.001    4.508    0.020 adam.py:347(_single_tensor_adam)
     1760    4.276    0.002    4.276    0.002 {method 'sqrt' of 'torch._C.TensorBase' objects}
249701/36266    0.332    0.000    0.612    0.000 deep_cfr.py:157(_traverse_batched)
    33266    0.016    0.000    0.604    0.000 {method 'send' of 'generator' objects}
      234    0.021    0.000    0.329    0.001 naive_deep_cfr.py:490(linear_cfr_loss)
     1095    0.004    0.000    0.303    0.000 activation.py:139(forward)
     1095    0.007    0.000    0.299    0.000 functional.py:1760(relu)
     1095    0.289    0.000    0.289    0.000 {built-in method torch.relu}
       42    0.003    0.000    0.241    0.006 naive_deep_cfr.py:524(_masked_softmax)
       42    0.235    0.006    0.235    0.006 {built-in method torch.softmax}
     4746    0.018    0.000    0.178    0.000 naive_deep_cfr.py:745(_require_finite_tensor)


```
