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
        self.lin1 = nn.Linear(latent_dim, 160)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(20, 8, 4, stride=2)  # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(8, 8, 4, stride=2)  # [n, 16, 34, 34]
        self.conv = nn.Conv2d(8, 8, kernel_size=10)  # [n, 8, 20, 20]

    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        print(f"x {x.shape}")
        print(f"A: {self.lin1.weight.shape}")
        x = self.lin1(x.T)
        x = F.relu(x)
        x.shape
        x = x.view(-1, 20, 8)  # 256  # 20, 8, 1

        # Upsample (transposed conv) 16x16 (64 feature maps)
        x = self.ct1(x)
        x = F.relu(x)

        # Upsample to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)
        print(x.shape)
        # Convolution to 28x28 (1 feature map)
        return self.conv(x)



class GAN(pl.LightningModule):
    def __init__(self, latenr_dim, lr = 0.002):
        super().__init__()
        self.save_hyperparameters()
        self.ldim =latenr_dim
        self.generator = Generator(self.hparams.latenr_dim)
        self.discriminator = Discriminator()

        # random noise
        self.validation_z = torch.randn(8, self.hparams.latenr_dim) # check

    def random_noise(self):
        return torch.randn(self.ldim, 8)

    def forward(self, z):
       return self.generator(z)

    def advarserial_loss(self,y_hat, y):
        return F.binary_cross_entropy((y_hat, y))

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs  = batch
        z = self.random_noise()
        print(z.shape)
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

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []



latenr_Dim = 20*8


true_signal  =getSignalPerCandidate()

trainer = pl.Trainer(max_epochs=20)
model = GAN(latenr_Dim)

trainer.fit(model, true_signal)