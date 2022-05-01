import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import scipy.io
import os
import sys
import pytorch_lightning as pl

open(os.path.join(os.path.dirname(sys.argv[0]), 'preprocessing.py'))
open(os.path.join(os.path.dirname(sys.argv[0]), 'Utilis.py'))

from preprocessing import *
from Utilis import *

Z_dim = 8


device = ''
BATCH_SIZE = 10 # Not sure whether this is cuorrect
# TODO: Check whetehr this should be 10


def tLoadeer(dataset ):
    transform = transforms.ToTensor()
    print("In")
    # trainData = torchvision.datasets.MNIST('./data/', download=True, transform=transform, train=True)
    # trainLoader = torch.utils.data.DataLoader(trainData, shuffle=True, batch_size=mb_size)
    return getSignalPerCandidate()
    # else:
    #     trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    #                                               pin_memory=True)
    # return trainLoader

# Simple Models
# class Gen(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(Z_dim, H_dim),
#             nn.ReLU(),
#             nn.Linear(H_dim, X_dim),
#             nn.Sigmoid()
#             # TODO: Checl whether wh should deepend the model
#         )
#
#     def forward(self, input):
#         return self.model(input)
#

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten the tensor so it can be fed into the FC layers
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7 * 7 * 64)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2)  # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2)  # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 8, kernel_size=15)  # [n, 8, 20, 20]

    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)  # 256

        # Upsample (transposed conv) 16x16 (64 feature maps)
        x = self.ct1(x)
        x = F.relu(x)

        # Upsample to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)

        # Convolution to 28x28 (1 feature map)
        return self.conv(x)



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
    # G = Gen().to(device)
    G = Generator(20*20)
    G =G.to('cpu')
    D = Dis().to('cpu')
    g_opt = opt.Adam(G.parameters(), lr=lr)
    d_opt = opt.Adam(D.parameters(), lr=lr)
    return G, D, g_opt, d_opt

class GAN(pl.LightningModule):
    def __init__(self, latenr_dim, lr = 0.002):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latenr_dim = latenr_dim)
        self.discriminator = Discriminator()

        # random noise
        self.validation_z = torch.randn(1, self.hpparams.latent_dim) # check

    def random_noise(self):
        return torch.randn(1, self.hpparams.latent_dim)

    def forward(self, z):
       return self.generator(z)

    def advarserial_loss(self,y_hat, y):
        return F.binary_cross_entropy((y_hat, y))

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs ,_ = batch
        z = self.random_noise()
        if optimizer_idx ==0:
            """ 
            train the generator
            """
            fake_img = self(z)
            # Get prediction
            y_hat = self.discriminator(fake_img)
            y = torch.ones(real_imgs.size(0), 1)

            g_loss = self.advarserial_loss(y_hat, y)
            log_dict = {"dictLoss": g_loss}
            return {"loss": g_loss, "progress_bar": log_dict, "log": log_dict}

        if optimizer_idx ==1:
            """ 
            miximizing the log (D(x) + log(1- D(G(z)))
            """
            y_hat_real = self.discriminator(real_imgs)
            y_real = torch.ones(real_imgs.size(0),1)
            real_loss = self.advarserial_loss(y_hat_real, y_real)

            y_hat_fake = self.discriminator(self(z).detach())
            y_fake = torch.zeros(real_imgs.size(0))
            y_fake_loss = self.advarserial_loss(y_hat_fake, y_fake)

            # calculating the loss together
            d_loss = (y_fake_loss + real_loss) /2

            log_dict = {"dictLoss": d_loss}
            return {"loss": d_loss, "progress_bar": log_dict, "log": log_dict}









        # Sample Noise
        z = torch.randn(real_imgs[0])


    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr = lr)
        opt_d = torch.optim.Adam(self.discriminator, lr = lr)
        return [opt_g, opt_d],[]

    def ofEpochend(self):
        self.plot_img






def trainModel(G, D, g_opt, d_opt, epoch):
    for epoch in range(epoch):
        G_loss_run = 0.0
        D_loss_run = 0.0
        for i, data in enumerate(trainLoader):
            X = data
            print(type(X))
            X = torch.from_numpy(X)
            print(type(X))
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
    # device = get_device()
    device = 'cpu'
    print(device)
    lr = 1e-3
    mb_size = 64

    trainLoader = tLoadeer(False)

    true_signal  =getSignalPerCandidate()
    input_dim, dim_n, dim_m =true_signal.shape

    dataIter = iter(true_signal)
    # signal = dataIter.next()

    Z_dim = dim_n * dim_m
    H_dim = 30
    X_dim = dim_n * dim_m
    #
    # G, D , g_opt, d_opt = init_opt()
    G = Generator(20*20)
    G =G.to('cpu')
    D = Dis().to('cpu')
    g_opt = opt.Adam(G.parameters(), lr=lr)
    d_opt = opt.Adam(D.parameters(), lr=lr)
    # print(Z_dim, H_dim, X_dim)
    #
    trainModel(G, D, g_opt, d_opt, 10)

    print(G)
    print("Finish")


