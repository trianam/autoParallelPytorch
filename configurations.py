from conf import Conf

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

configTestOptConf = lambda param: Conf({
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
    "path":                 'configTestOpt/{}-{}-{}'.format(param['nf1'],param['nf2'],param['hd']),
    "modelLoad":            'best.pt',
    })

configTestOptParam = [{
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

configTestOpt = {
    "conf":         configTestOptConf,
    "param":        configTestOptParam,
    "saveFile":     "files/configTestOpt/hyperparametersOpt.p",
    "trials":       20,
    "checkMetric":  "accuracy",
}