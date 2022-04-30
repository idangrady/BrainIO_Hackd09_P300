import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

device = ''
BATCH_SIZE = 10 # Not sure whether this is cuorrect
# TODO: Check whetehr this should be 10


def tLoadeer(dataset = False ):
    transform = transforms.ToTensor()
    if (dataset) :
        trainData = torchvision.datasets.MNIST('./data/', download=True, transform=transform, train=True)
        trainLoader = torch.utils.data.DataLoader(trainData, shuffle=True, batch_size=mb_size)
    else:
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                                  pin_memory=True)
    return trainLoader

# Simple Models
class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(Z_dim, H_dim),
            nn.ReLU(),
            nn.Linear(H_dim, X_dim),
            nn.Sigmoid()
            # TODO: Checl whether wh should deepend the model
        )

    def forward(self, input):
        return self.model(input)

class Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(X_dim, H_dim),
            nn.ReLU(),
            nn.Linear(H_dim, 1),
            nn.Sigmoid()
            # TODO: Check whether we should add more layers
        )
#
# def initModel():
#     return (Dis(),Gen() )

    def forward(self, input):
        return self.model(input)

def init_opt(G, D):
    g_opt = opt.Adam(G.parameters(), lr=lr)
    d_opt = opt.Adam(D.parameters(), lr=lr)

if __name__ == "__main__":

    lr = 1e-3
    mb_size = 64

    trainLoader = tLoadeer()

    dataIter = iter(trainLoader)
    imgs, labels = dataIter.next()

    Z_dim = 100
    H_dim = 128
    X_dim = imgs.view(imgs.size(0), -1).size(1)

    print(Z_dim, H_dim, X_dim)


