# torch.profiler: optimised Leduc Deep CFR

This automatically generated diagnostic records PyTorch operator CPU time. It uses a separate warmed run and is not a formal runtime measurement.

- Outer iterations: 1
- Traversals: 500
- Optimizer steps: 60

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                               aten::mm        33.95%        2.485s        33.96%        2.486s       5.918ms           420  
                                            aten::addmm        19.99%        1.463s        20.40%        1.493s       5.332ms           280  
                                              aten::sum        17.42%        1.275s        17.44%        1.277s       3.432ms           372  
                                             aten::sqrt        15.73%        1.151s        15.73%        1.151s       2.399ms           480  
                                            aten::index         5.29%     387.224ms         5.31%     388.873ms       1.200ms           324  
                                         aten::_softmax         1.67%     122.474ms         1.67%     122.474ms       5.567ms            22  
                           aten::_softmax_backward_data         1.50%     110.151ms         1.50%     110.151ms       5.508ms            20  
                               Optimizer.step#Adam.step         0.68%      49.954ms        16.87%        1.235s      20.584ms            60  
                                        aten::clamp_min         0.53%      38.973ms         0.53%      38.973ms     185.586us           210  
                                            aten::copy_         0.48%      34.869ms         0.48%      34.869ms       8.090us          4310  
    autograd::engine::evaluate_function: AddmmBackward0         0.24%      17.552ms        51.98%        3.805s      15.854ms           240  
                                              aten::abs         0.23%      16.870ms         0.38%      27.861ms      10.109us          2756  
                                         AddmmBackward0         0.19%      13.620ms        34.29%        2.510s      10.458ms           240  
                                                aten::t         0.15%      10.862ms         0.28%      20.162ms      17.086us          1180  
                               aten::threshold_backward         0.12%       9.019ms         0.12%       9.019ms      50.108us           180  
                                              aten::all         0.11%       8.334ms         0.13%       9.742ms       7.391us          1318  
                                              aten::div         0.11%       7.929ms         0.20%      14.960ms      24.686us           606  
                                               aten::ne         0.09%       6.906ms         0.18%      13.516ms       9.808us          1378  
                                         aten::_to_copy         0.08%       6.138ms         0.24%      17.288ms       4.477us          3862  
                                       aten::as_strided         0.08%       5.858ms         0.08%       5.858ms       1.337us          4380  
                                           aten::linear         0.08%       5.541ms        20.56%        1.505s       5.374ms           280  
                                        aten::transpose         0.08%       5.506ms         0.13%       9.300ms       7.881us          1180  
                                             aten::add_         0.07%       5.489ms         0.11%       8.286ms       8.632us           960  
                                    aten::empty_strided         0.07%       5.149ms         0.07%       5.149ms       1.297us          3970  
                                              aten::mul         0.06%       4.702ms         0.08%       5.896ms       3.246us          1816  
                                               aten::eq         0.06%       4.238ms         0.06%       4.508ms       3.271us          1378  
                                         aten::isfinite         0.06%       4.176ms         0.58%      42.386ms      32.160us          1318  
                                             aten::relu         0.06%       4.042ms         0.59%      43.015ms     204.833us           210  
                                             aten::view         0.05%       3.927ms         0.05%       3.927ms       6.233us           630  
                                         aten::addcdiv_         0.04%       3.192ms         0.04%       3.192ms       6.650us           480  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 7.320s

```
