import torch
import numpy as np
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from filelock import FileLock
import os

def mnist(batch_size, test=False):
    with FileLock(os.path.expanduser("~/data.lock")):
        trainVal = datasets.MNIST('data', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

        myDatasets = {
            'test': datasets.MNIST('data', train=False,
                                   transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ])),
        }

        indices = np.array(range(len(trainVal)))
        np.random.shuffle(indices)
        validIndices = indices[:len(myDatasets['test'])]
        trainIndices = indices[len(myDatasets['test']):]
        myDatasets['valid'] = Subset(trainVal, validIndices)
        myDatasets['train'] = Subset(trainVal, trainIndices)

        myLoaders = {s: torch.utils.data.DataLoader(myDatasets[s], batch_size=batch_size, shuffle=True) for s in myDatasets}

    return myLoaders, myDatasets
