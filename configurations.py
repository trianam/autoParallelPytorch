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
        "modelSave":            "best",
        "path":                 'configTest',
        "modelLoad":            'last.pt',
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
        "epochs":               1,#14,
        "tensorBoard":          True,
        "modelSave":            "best",
        "path":                 'configTestParallel/{}'.format(i),
        "modelLoad":            'last.pt',
    })
