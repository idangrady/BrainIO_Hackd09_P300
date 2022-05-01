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


open(os.path.join(os.path.dirname(sys.argv[0]), 'GANLowLevel.py'))
from GANLowLevel import GAN


latenr_Dim = 20*8

trainer = pl.Trainer(max_epochs=20)
model = GAN()
trainer.fit(model, )