# -*- coding: utf-8 -*-
"""
## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@durham.ac.uk

This notebook will provide an example that shows the implementation of a simple AutoEncoder in PyTorch.

Copyright (c) 2023 Amir Atapour-Abarghouei, UK.

based on: https://colab.research.google.com/gist/cwkx/e3ef25d0adb6e2f2bf747ce664bab318/conv-autoencoder.ipynb#scrollTo=RGbLY6X-NH4O

License : LGPL - http://www.gnu.org/licenses/lgpl.html

We are going to implement an autoEncoder. Let's start by importing what we need:
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device is {device}!")

"""Let's set a few variables to make things easier:"""

batch_size  = 256
n_channels  = 3
latent_size = 512
print('done!')

"""We should now line up the dataset we are going to use. We will be working with the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

To process the data, we are going to convert the images to 32x32 images.
"""

train_dataset = torchvision.datasets.CIFAR10(
    'data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor()
]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size, drop_last=True
)

print('\nThe classes are:')
print(*train_dataset.classes, sep = ", ")

"""Now that the dataset is ready, let's look at a few of our images:"""

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_loader.dataset[i][0].permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.xlabel(train_dataset.classes[train_loader.dataset[i][1]])

"""We can now create our model, which will be a very simple convolutional network:"""

# simple block of convolution, batchnorm, and leakyrelu
class Block(nn.Module):
    def __init__(self, in_f, out_f):
        super(Block, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_f),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.f(x)

# define the model
class Autoencoder(nn.Module):
    def __init__(self, f=16):
        super().__init__()

        self.encode = nn.Sequential(
            Block(n_channels, f),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 16x16
            Block(f  ,f*2),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 8x8
            Block(f*2,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 4x4
            Block(f*4,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 2x2
            Block(f*4,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 1x1
            Block(f*4,latent_size),
        )

        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2), # output = 2x2
            Block(latent_size,f*4),
            nn.Upsample(scale_factor=2), # output = 4x4
            Block(f*4,f*4),
            nn.Upsample(scale_factor=2), # output = 8x8
            Block(f*4,f*2),
            nn.Upsample(scale_factor=2), # output = 16x16
            Block(f*2,f  ),
            nn.Upsample(scale_factor=2), # output = 32x32
            nn.Conv2d(f,n_channels, 3,1,1),
            nn.Sigmoid()
        )

A = Autoencoder().to(device)
print(f'The model has {len(torch.nn.utils.parameters_to_vector(A.parameters()))} parameters.')

print('The optimiser has been created!')
optimiser = torch.optim.Adam(A.parameters(), lr=0.001)

"""Let's start the main training loop now:"""

epoch = 0
# training loop
while (epoch < 20):
    
    # for metrics
    loss_arr = np.zeros(0)

    # iterate over the training dateset
    for i, batch in enumerate(train_loader):

        # sample x from the dataset
        x, t = batch
        x, t = x.to(device), t.to(device)

        # do the forward pass with mean squared error
        z = A.encode(x)
        x_hat = A.decode(z)
        # calculate loss:
        loss = ((x-x_hat)**2).mean()

        # backpropagate to compute the gradients of the loss w.r.t the parameters and optimise
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # collect stats
        loss_arr = np.append(loss_arr, loss.item())

    # sample
    z = torch.randn_like(z)
    g = A.decode(z)

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

    epoch = epoch+1