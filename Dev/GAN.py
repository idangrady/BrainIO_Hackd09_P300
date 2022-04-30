import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from Utilis import *
# from .../dataLoadInDataset import
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

device = ''
BATCH_SIZE = 10 # Not sure whether this is cuorrect
# TODO: Check whetehr this should be 10


def tLoadeer(dataset ):
    transform = transforms.ToTensor()
    print("In")
    trainData = torchvision.datasets.MNIST('./data/', download=True, transform=transform, train=True)
    trainLoader = torch.utils.data.DataLoader(trainData, shuffle=True, batch_size=mb_size)
    return trainLoader
    # else:
    #     trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    #                                               pin_memory=True)
    # return trainLoader

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

    def forward(self, input):
        return self.model(input)

def init_opt():
    device = 'cpu'
    G = Gen().to(device)
    D = Dis().to(device)
    g_opt = opt.Adam(G.parameters(), lr=lr)
    d_opt = opt.Adam(D.parameters(), lr=lr)
    return G, D, g_opt, d_opt


def trainModel(G, D, g_opt, d_opt, epoch):
    for epoch in range(epoch):
        G_loss_run = 0.0
        D_loss_run = 0.0
        for i, data in enumerate(trainLoader):
            X, _ = data
            X = X.view(X.size(0), -1).to(device)
            mb_size = X.size(0)

            one_labels = torch.ones(mb_size, 1).to(device)
            zero_labels = torch.zeros(mb_size, 1).to(device)

            z = torch.randn(mb_size, Z_dim).to(device)

            D_real = D(X)
            D_fake = D(G(z))

            D_real_loss = F.binary_cross_entropy(D_real, one_labels)
            D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels)
            D_loss = D_real_loss + D_fake_loss

            d_opt.zero_grad()
            D_loss.backward()
            d_opt.step()

            z = torch.randn(mb_size, Z_dim).to(device)
            D_fake = D(G(z))
            G_loss = F.binary_cross_entropy(D_fake, one_labels)

            g_opt.zero_grad()
            G_loss.backward()
            g_opt.step()

            G_loss_run += G_loss.item()
            D_loss_run += D_loss.item()

        print('Epoch:{},   G_loss:{},    D_loss:{}'.format(epoch, G_loss_run / (i + 1), D_loss_run / (i + 1)))

        samples = G(z).detach()
        samples = samples.view(samples.size(0), 1, 28, 28).cpu()


if __name__ == "__main__":
    device = get_device()
    print(device)
    lr = 1e-3
    mb_size = 64

    trainLoader = tLoadeer(False)

    dataIter = iter(trainLoader)
    imgs, labels = dataIter.next()

    Z_dim = 100
    H_dim = 128
    X_dim = imgs.view(imgs.size(0), -1).size(1)


    G, D , g_opt, d_opt = init_opt()
    print(Z_dim, H_dim, X_dim)

    trainModel(G, D, g_opt, d_opt, 10)

    print("Finish")


