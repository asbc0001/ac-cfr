# torch.profiler: reference Leduc Deep CFR

This automatically generated diagnostic records PyTorch operator CPU time. It uses a separate warmed run and is not a formal runtime measurement.

- Outer iterations: 1
- Traversals: 500
- Optimizer steps: 60

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                               aten::mm        33.64%        2.478s        33.64%        2.479s       5.902ms           420  
                                            aten::addmm        18.37%        1.353s        18.82%        1.387s     325.190us          4264  
                                              aten::sum        17.05%        1.256s        17.08%        1.258s       3.383ms           372  
                                             aten::sqrt        16.66%        1.227s        16.66%        1.227s       2.557ms           480  
                                            aten::index         5.60%     412.390ms         5.61%     413.625ms       1.277ms           324  
                                         aten::_softmax         2.15%     158.662ms         2.15%     158.662ms       7.212ms            22  
                           aten::_softmax_backward_data         1.97%     144.968ms         1.97%     144.968ms       7.248ms            20  
                               Optimizer.step#Adam.step         0.72%      52.850ms        17.78%        1.310s      21.832ms            60  
                                        aten::clamp_min         0.50%      36.910ms         0.50%      36.910ms      11.542us          3198  
                                            aten::copy_         0.49%      35.917ms         0.49%      35.917ms       4.333us          8290  
    autograd::engine::evaluate_function: AddmmBackward0         0.24%      18.004ms        51.37%        3.785s      15.770ms           240  
                                         AddmmBackward0         0.24%      17.425ms        34.03%        2.507s      10.446ms           240  
                                              aten::abs         0.17%      12.890ms         0.30%      22.445ms       8.168us          2748  
                                                aten::t         0.17%      12.176ms         0.32%      23.585ms       4.567us          5164  
                               aten::threshold_backward         0.14%      10.625ms         0.14%      10.625ms      59.029us           180  
                                       aten::as_strided         0.12%       8.829ms         0.12%       8.829ms       0.509us         17344  
                                              aten::all         0.10%       7.507ms         0.12%       8.860ms       6.742us          1314  
                                              aten::div         0.09%       6.795ms         0.17%      12.558ms      20.723us           606  
                                               aten::ne         0.09%       6.485ms         0.16%      12.070ms       8.785us          1374  
                                        aten::transpose         0.09%       6.326ms         0.15%      11.409ms       2.209us          5164  
                                         aten::_to_copy         0.08%       5.816ms         0.20%      14.960ms       3.878us          3858  
                                             aten::relu         0.07%       5.431ms         0.57%      42.341ms      13.240us          3198  
                                           aten::linear         0.07%       5.281ms        19.03%        1.402s     328.866us          4264  
                                              aten::mul         0.07%       4.860ms         0.08%       6.159ms       3.399us          1812  
                                         aten::isfinite         0.06%       4.127ms         0.50%      36.529ms      27.800us          1314  
                                               aten::eq         0.05%       3.987ms         0.06%       4.231ms       3.079us          1374  
                                             aten::add_         0.05%       3.962ms         0.09%       6.805ms       7.088us           960  
                                             aten::view         0.05%       3.909ms         0.05%       3.909ms       6.204us           630  
                                    aten::empty_strided         0.05%       3.731ms         0.05%       3.731ms       0.941us          3966  
                                         aten::addcdiv_         0.04%       3.156ms         0.04%       3.156ms       6.576us           480  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 7.368s

```
