# autoParallelPytorch
Try to run multiple istances of pytorch train untill it fills the memory. When the memory is full wait few minutes and retry.
## instructions
1. copy `notifierConfigTEMPLATE.py` to `notifierConfig.py` and modify it with mail configuration (or comment `notifier` in the runners).
2. to run:
   * `python runPytorch.py configTest` for single process;
   * `python runPytorchParallel.py configTestParallel` for multiprocess;
   * `python runPytorchAx.py configTestAx` for hyperparameter optimixation with Ax in single process.
   * `python runPytorchTune.py configTestTune` for grid search with Tune in multitask