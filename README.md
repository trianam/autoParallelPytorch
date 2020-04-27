# autoParallelPytorch
Try to run multiple istances of pytorch train untill it fills the memory. When the memory is full wait few minutes and retry.
## Instructions
1. copy `notifierConfigTEMPLATE.py` to `notifierConfig.py` and modify it with mail configuration (or comment `notifier` in the runners).
2. to run:
   * `python runPytorch.py configTest` for single process;
   * `python runPytorchParallel.py configTestParallel` for multiprocess;
   * `python runPytorchAx.py configTestAx` for hyperparameter optimixation with Ax in single process.
   * `python runPytorchTune.py configTestTune` for grid search with Tune in multitask
## References
* https://ray.readthedocs.io/en/latest/tune.html
* https://ax.dev/
* https://towardsdatascience.com/fast-hyperparameter-tuning-at-scale-d428223b081c
* https://towardsdatascience.com/rocking-hyperparameter-tuning-with-pytorchs-ax-package-1c2dd79f2948
* https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f
* http://hyperopt.github.io/hyperopt/
* https://en.wikipedia.org/wiki/Hyperparameter_optimization
