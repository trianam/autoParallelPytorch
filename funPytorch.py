import os
import time
import warnings

import torch
import torch.nn.functional as F

import torchsummary
from tensorboardX import SummaryWriter

from collections import defaultdict

import loaders
import model1


def makeModel(conf, device):
    if conf.model == 'model1':
        model = model1.MyModel(conf)
    else:
        raise Exception("Model don't exists")

    model = model.to(device)

    if conf.optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=conf.learningRate, weight_decay=conf.learningRateDecay)
    elif conf.optimizer == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=conf.learningRate, weight_decay=conf.learningRateDecay)

    return model,optim

def summary(conf, model):
    torchsummary.summary(model, (conf.batchSize, conf.numFields, conf.numPhrases, conf.phraseLen, conf.vecLen))

def processData(conf):
    if conf.dataset == 'mnist':
        return loaders.mnist(conf.batchSize)

def evaluate(conf, model, dataloader):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        runningLoss = 0.
        runningMetrics = defaultdict(float)

        for data in dataloader:
            X,y = data
            X,y = X.to(device), y.to(device)

            yp = model(X)

            loss = F.nll_loss(yp, y)

            runningLoss += loss.item()

            pred = yp.argmax(dim=1, keepdim=True)
            runningMetrics['accuracy'] += pred.eq(y.view_as(pred)).sum().item()

        runningMetrics['accuracy'] /= len(dataloader.dataset)

        return [(runningLoss / len(dataloader)), runningMetrics]

def runTrain(conf, model, optim, dataloaders, verbose=False):
    trainDataloader = dataloaders['train']
    testDataloader = dataloaders['test']

    device = next(model.parameters()).device

    if conf.tensorBoard:
        writer = SummaryWriter(os.path.join("files",conf.path,"tensorBoard"), flush_secs=60)

    if not os.path.exists(os.path.join("files",conf.path,"models")):
        os.makedirs(os.path.join("files",conf.path,"models"))

    for epoch in range(conf.startEpoch, conf.startEpoch+conf.epochs):
        if verbose:
            print("epoch {}".format(epoch), end='', flush=True)

        model.train()
        for batchIndex, data in enumerate(trainDataloader):
            X,y = data
            X,y = X.to(device), y.to(device)

            model.zero_grad()
            yp = model(X)

            loss = F.nll_loss(yp, y)
            loss.backward()
            optim.step()

        if verbose:
            print(": ", end='', flush=True)
        # else:
        #     print(".", end='', flush=True)

        trainLoss, trainMetrics = evaluate(conf, model, trainDataloader)
        testLoss, testMetrics = evaluate(conf, model, testDataloader)

        if verbose:
            print("train loss {}; test loss {}".format(trainLoss, testLoss), end='', flush=True)
            for k in trainMetrics:
                print("; train {} {}".format(k, trainMetrics[k]), end='', flush=True)
            for k in testMetrics:
                print("; test {} {}".format(k, testMetrics[k]), end='', flush=True)
            print("", flush=True)

        writerDictLoss = {
            'train': trainLoss,
            'test': testLoss,
            }

        writerDictMetrics = {}
        for k in trainMetrics: #same keys for train and valid
            writerDictMetrics[k] = {
                'train': trainMetrics[k],
                'test': testMetrics[k],
                }

        #save always last
        fileLast = os.path.join("files",conf.path,"models","last.pt")

        if os.path.isfile(fileLast):
            os.remove(fileLast)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'epoch': epoch
        }, fileLast)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if conf.tensorBoard:
                writer.add_scalars('loss', writerDictLoss, epoch)

                for k in writerDictMetrics:
                    writer.add_scalars(k, writerDictMetrics[k], epoch)

    time.sleep(120) #time to write tensorboard

