from conf import Conf
from ray import tune
import numpy as np

configTest = Conf({
    "optimizer":            'adam',
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "model":                'model1',
    "numFilters1":          32,
    "numFilters2":          64,
    "hiddenDim":            128,
    "dataset":              'mnist',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "patience":             10,
    "tensorBoard":          True,
    "checkMetric":          "accuracy",
    "path":                 'configTest',
    "modelLoad":            'best.pt',
    })

configTestParallel = {}
for i in range(15):
    configTestParallel[i] = Conf({
        "optimizer":            'adam',
        "learningRate":         0.001,
        "learningRateDecay":    0.,
        "model":                'model1',
        "numFilters1":          32,
        "numFilters2":          64,
        "hiddenDim":            128,
        "dataset":              'mnist',
        "batchSize":            64,
        "startEpoch":           0,
        "epochs":               100,
        "patience":             10,
        "tensorBoard":          True,
        "checkMetric":          "accuracy",
        "path":                 'configTestParallel/{}'.format(i),
        "modelLoad":            'best.pt',
    })

configTestAxConf = lambda param: Conf({
    "optimizer":            'adam',
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "model":                'model1',
    "numFilters1":          param['nf1'],
    "numFilters2":          param['nf2'],
    "hiddenDim":            param['hd'],
    "dataset":              'mnist',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "patience":             10,
    "tensorBoard":          True,
    "checkMetric":          "accuracy",
    "path":                 'configTestAx/{}-{}-{}'.format(param['nf1'],param['nf2'],param['hd']),
    "modelLoad":            'best.pt',
    })

configTestAxParam = [{
        "name": "nf1",
        "type": "range",
        "bounds": [4, 128],
        "log_scale": True,
    },{
        "name": "nf2",
        "type": "range",
        "bounds": [4, 128],
        "log_scale": True,
    },{
        "name": "hd",
        "type": "range",
        "bounds": [4, 256],
        "log_scale": True,
    }]

configTestAx = {
    "conf":         configTestAxConf,
    "param":        configTestAxParam,
    "saveFile":     "files/configTestAx/hyperparametersOpt.p",
    "trials":       20,
    "checkMetric":  "accuracy",
}

configTestTuneConf = lambda param: Conf({
    "optimizer":            'adam',
    "learningRate":         0.001,
    "learningRateDecay":    0.,
    "model":                'model1',
    "numFilters1":          param['nf1'],
    "numFilters2":          param['nf2'],
    "hiddenDim":            param['hd'],
    "dataset":              'mnist',
    "batchSize":            64,
    "startEpoch":           0,
    "epochs":               100,
    "patience":             10,
    "tensorBoard":          False,
    "checkMetric":          "accuracy",
    "path":                 'configTestTune/{}-{}-{}'.format(param['nf1'],param['nf2'],param['hd']),
    "modelLoad":            'best.pt',
    })

configTestTuneParam = {
        "nf1": tune.grid_search(list(np.logspace(4,7,7-4+1,base=2, dtype=np.int))),
        "nf2":  tune.grid_search(list(np.logspace(4,7,7-4+1,base=2, dtype=np.int))),
        "hd": tune.grid_search(list(np.logspace(5,8,8-5+1,base=2, dtype=np.int))),
        }

configTestTune = Conf({
    "conf":             configTestTuneConf,
    "param":            configTestTuneParam,
    "resultsDir":       "files/configTestTune/ray_results",
    "saveFile":         "files/configTestTune/hyperparametersOpt.p",
    "trialResources":   {'cpu':1,'gpu':1./10},
})