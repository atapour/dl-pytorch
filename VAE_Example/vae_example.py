# -*- coding: utf-8 -*-
"""
# Deep Learning - Variational Autoencoder (VAE) in PyTorch
---

## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@durham.ac.uk

This notebook will provide an example that shows the implementation of a simple Variational Autoencoder (VAE) in PyTorch.

Copyright (c) 2022 Amir Atapour-Abarghouei, UK.

License : LGPL - http://www.gnu.org/licenses/lgpl.html

We are going to implement an autoencoder. Let's start by importing what we need:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device is {device}!")

"""We should now line up the dataset we are going to use. We will be working with the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset."""

train_dataset = torchvision.datasets.MNIST(
    'data', train=True, download=True, transform=torchvision.transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=256, drop_last=True
)

print('\nThe classes are:')
print(*train_dataset.classes, sep = ", ")
print(f'There are {len(train_dataset)} images in the training set.')

"""Now that the dataset is ready, let's look at a few of our images in our training set:"""

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_loader.dataset[i][0].clamp(0,1).repeat(3,1,1).permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.xlabel(train_dataset.classes[train_loader.dataset[i][1]])

"""We can now create our model, which will be a very simple convolutional network:"""

# define the model
class VAE(nn.Module):
    def __init__(self, n_channels=1, f_dim=32*20*20, z_dim=256):
        super().__init__()

        # encoder layers:
        self.enc_conv1 = nn.Conv2d(n_channels, 16, 5)
        self.enc_conv2 = nn.Conv2d(16, 32, 5)
        # two linear layers with one for the mean and the other the variance
        self.enc_linear1 = nn.Linear(f_dim, z_dim)
        self.enc_linear2 = nn.Linear(f_dim, z_dim)

        # decoder layers:
        self.dec_linear = nn.Linear(z_dim, f_dim)
        self.dec_conv1 = nn.ConvTranspose2d(32, 16, 5)
        self.dec_conv2 = nn.ConvTranspose2d(16, n_channels, 5)

    # encoder:
    def encoder(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(-1, 32*20*20)
        # the output is mean (mu) and variance (logVar)
        mu = self.enc_linear1(x)
        logVar = self.enc_linear2(x)
        # mu and logVar are used to sample z and compute KL divergence loss
        return mu, logVar

    # reparameterisation trick:
    def reparameterise(self, mu, logVar):
        # from mu and logVar, we can sample via mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    # decoder:
    def decoder(self, z):
        x = F.relu(self.dec_linear(z))
        x = x.view(-1, 32, 20, 20)
        x = F.relu(self.dec_conv1(x))
        # the output is the same size as the input
        x = torch.sigmoid(self.dec_conv2(x))
        return x

    # forward pass:
    def forward(self, x):
        mu, logVar = self.encoder(x)
        z = self.reparameterise(mu, logVar)
        out = self.decoder(z)
        # mu and logVar are returned as well as the output for loss computation
        return out, mu, logVar

model = VAE().to(device)
print(f'The model has {len(torch.nn.utils.parameters_to_vector(model.parameters()))} parameters.')

print('The optimiser has been created!')
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

"""Let's start the main training loop now:"""

epoch = 0
# training loop
while (epoch < 20):
    
    # for metrics
    loss_arr = np.zeros(0)

    # iterate over the training dateset
    for i, batch in enumerate(train_loader):

        # sample x from the dataset
        x, _ = batch
        x = x.to(device)

        # forward pass to obtain image, mu, and logVar
        x_hat, mu, logVar = model(x)

        # caculate loss - BCE combined with KL
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = F.binary_cross_entropy(x_hat, x, size_average=False) + kl_divergence

        # backpropagate to compute the gradients of the loss w.r.t the parameters and optimise
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # collect stats
        loss_arr = np.append(loss_arr, loss.item())

    # sample
    z = torch.randn_like(mu)
    g = model.decoder(z)

    # plot some examples from training
    print("\n============================================")
    print(f'Epoch {epoch} Loss: {loss.mean().item()}')
    print('Training Examples')
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(x_hat[:16]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()

    # plot results sampled from latent
    print('Samples from the Latent Space')
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(g[:16]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()
    print("============================================\n")

    epoch += 1