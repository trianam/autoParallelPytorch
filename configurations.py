from conf import Conf

configTest = Conf({
        "optimizer":            'adam',
        "learningRate":         0.001,
        "learningRateDecay":    0.,
        "model":                'model1',
        "dataset":              'mnist',
        "batchSize":            64,
        "startEpoch":           0,
        "epochs":               14,
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
        "dataset":              'mnist',
        "batchSize":            64,
        "startEpoch":           0,
        "epochs":               14,
        "tensorBoard":          True,
        "checkMetric":          "accuracy",
        "path":                 'configTestParallel/{}'.format(i),
        "modelLoad":            'best.pt',
    })
