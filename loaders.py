import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

def mnist(batch_size, test=False):
    myDatasets = {
        'train': datasets.MNIST('data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ])),
        'test': datasets.MNIST('data', train=False,
                               transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ])),
    }

    myLoaders = {s: torch.utils.data.DataLoader(myDatasets[s], batch_size=batch_size, shuffle=True) for s in myDatasets}

    return myLoaders, myDatasets
