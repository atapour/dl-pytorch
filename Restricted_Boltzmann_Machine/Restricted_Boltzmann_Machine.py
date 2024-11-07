# -*- coding: utf-8 -*-
"""

# Deep Learning - Restricted Boltzmann Machine in PyTorch
---

## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@durham.ac.uk

This notebook will provide an example of a Restricted Boltzmann Machine in PyTorch.

Copyright (c) 2024 Amir Atapour-Abarghouei, UK.

License : LGPL - http://www.gnu.org/licenses/lgpl.html

Let's start by importing what we need:
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import sleep

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

"""Let's import the dataset. We will use the MNIST dataset for simplicity:"""

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
shuffle=True, batch_size=64, drop_last=True)

img_width = 28
n_channels = 1
train_iterator = iter(cycle(train_loader))
print(f'Size of training dataset: {len(train_loader.dataset)}')

"""Here, we will view some of the images in the dataset:"""

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if n_channels == 1:
        plt.imshow(train_loader.dataset[i][0].clamp(0,1).repeat(3,1,1).permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)

"""Now, let's create the Restricted Boltzmann machine model and sampling functions."""

# https://github.com/odie2630463/Restricted-Boltzmann-Machines-in-pytorch
class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hin=500, k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin,n_vis).to(device)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis).to(device))
        self.h_bias = nn.Parameter(torch.zeros(n_hin).to(device))
        self.k = k

    def sample_from_p(self,p):
        # samples are conditionally independeant, so we can sample from univariate random variables
        return F.relu(torch.sign(p - (torch.rand(p.size()).to(device))))

    def v_to_h(self,v):
        p_h = torch.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h

    def h_to_v(self,h):
        p_v = torch.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v

    def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)

        h_ = h1
        for _ in range(self.k): # MCMC approximation, in practice k=1 is used
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)

        return v,v_

    def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

R = RBM(k=1).to(device)
print(f'Number of model parameters: {len(torch.nn.utils.parameters_to_vector(R.parameters()))}')

# initialise the optimiser
optimiser = torch.optim.Adam(R.parameters(), lr=0.0002, betas=(0.5, 0.999))
epoch = 0

"""Now, we can start with the main training loop:"""

# training loop
while (epoch<20):

    # arrays for metrics
    loss_arr = np.zeros(0)

    # iterate over some of the train dateset
    for i in range(500):
        x,t = next(train_iterator)
        x,t = x.to(device), t.to(device)

        x = x.view(-1,784)
        sample_data = x.bernoulli().to(device)

        v,v1 = R(sample_data)
        loss = R.free_energy(v) - R.free_energy(v1)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loss_arr = np.append(loss_arr, loss.item())

    # plot some examples
    g = v1.view(-1,1,28,28)
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(g).cpu().data.clamp(0,1).permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()
    plt.pause(0.0001)

    epoch = epoch+1