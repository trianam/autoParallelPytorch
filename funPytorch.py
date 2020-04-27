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

def runTrain(conf, model, optim, dataloaders, returnValidMetric=False, verbose=False):
    fileLast = os.path.join("files", conf.path, "models", "last.pt")
    fileBest = os.path.join("files", conf.path, "models", "best.pt")

    device = next(model.parameters()).device

    if conf.tensorBoard:
        writer = SummaryWriter(os.path.join("files",conf.path,"tensorBoard"), flush_secs=60)

    if not os.path.exists(os.path.join("files",conf.path,"models")):
        os.makedirs(os.path.join("files",conf.path,"models"))

    bestMetric = None
    bestMetricEpoch = 0
    for epoch in range(conf.startEpoch, conf.startEpoch+conf.epochs):
        if verbose:
            print("epoch {}".format(epoch), end='', flush=True)

        model.train()
        for batchIndex, data in enumerate(dataloaders['train']):
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

        losses = {}
        metrics = {}
        for d in dataloaders:
            losses[d],metrics[d] = evaluate(conf, model, dataloaders[d])

        metrics = {
            k: {
                d: metrics[d][k] for d in metrics
            } for k in metrics['train']}  # same keys for train, valid and test

        if verbose:
            first = True
            for d in losses:
                if not first:
                    print("; ", end='', flush=True)
                first = False
                print("{} loss {:.4f}".format(d, losses[d]), end='', flush=True)
            for k in metrics:
                for d in metrics[k]:
                    print("; {} {} {:.4f}".format(d, k, metrics[k][d]), end='', flush=True)
            print("", flush=True)

        #save always last
        if os.path.isfile(fileLast):
            os.remove(fileLast)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'epoch': epoch
        }, fileLast)

        #save best if necessary
        if bestMetric is None or metrics[conf.checkMetric]['valid'] > bestMetric:
            bestMetric = metrics[conf.checkMetric]['valid']
            bestMetricEpoch = epoch

            if os.path.isfile(fileBest):
                os.remove(fileBest)

            torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'epoch': epoch
            }, fileBest)


        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if conf.tensorBoard:
                writer.add_scalars('loss', losses, epoch)

                for k in metrics:
                    writer.add_scalars(k, metrics[k], epoch)

        if not conf.patience is None and epoch-bestMetricEpoch >= conf.patience:
            break

    time.sleep(120) #time to write tensorboard

    if returnValidMetric:
        return bestMetric

