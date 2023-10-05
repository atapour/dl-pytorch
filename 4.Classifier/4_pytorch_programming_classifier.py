# -*- coding: utf-8 -*-
"""
# PyTorch Programming - Demonstrating A Simple Classifier
---

## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@durham.ac.uk

This notebook will provide an example that shows the use of a simple neural network for classification in PyTorch.

Let's start by importing what we need:
"""

import torch
import torch.nn as nn
import torchvision

print('done!')

"""Since we are going to be training a model, we want to make sure that we are using a GPU to accelerate our training. If you are using Google Colab, you should:

Select Runtime -> Change runtime type -> GPU

If you are running this code on a local machine that has GPU hardware, you can just run:
"""

device = torch.device('cuda')
# device = torch.device('cpu')
print(f'Device type is {device}.')

"""We are going to use the [livelossplot](https://github.com/stared/livelossplot) library to help us plot our loss:"""

!pip install livelossplot --quiet

from livelossplot import PlotLosses

"""Here, we're going to define a helper function to make getting another batch of data easier."""

def cycle(iterable):
  while True:
    for x in iterable:
      yield x

print('done!')

"""For this example, we are going to be using the [FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html) dataset. [This dataset](https://github.com/zalandoresearch/fashion-mnist) consists of 60,000 images of items of clothing with the same resolution as the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

In the following, these are the categories of the data:
"""

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print("The classes are:", *class_names, sep = ", ")

"""Let's load the data for the training and test sets:"""

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor()
    ])),
shuffle=True, batch_size=256, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor()
    ])),
batch_size=512, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'Size of training set: {len(train_loader.dataset)}')
print(f'Size of test set: {len(test_loader.dataset)}')

"""We can have a look at some of the data:"""

import matplotlib.pyplot as plt

x, y = next(train_iterator)

grid = torchvision.utils.make_grid(x[:16])
grid = (grid-grid.min())/(grid.max()-grid.min())

plt.imshow(grid.permute(1, 2, 0))

print(x.shape)

"""Now that the data is ready to go, let's create a simple neural network. Our network consists of a linear layer with a [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) activation function followed by a second linear layer:"""

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.l1 = nn.Linear(in_features=1024, out_features=512)
        self.l2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)
        return x

model = SimpleNetwork().to(device)
print(model)

"""Our neural network should have its weights with the `requires_grad` already set and on the device we want it to be, whether it be CPU or GPU. Let's look at the weights of the first layer: """

print(model.l1.weight)

"""While we can manually adjust the parameters by running an optimisation algorithm like Stochastic Gradient Descent, PyTorch offers a variety of [optimisers](https://pytorch.org/docs/stable/optim.html#algorithms) that can help train our neural networks easily.

Here, we will use [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam) [from [this paper](https://arxiv.org/abs/1412.6980)], which takes advantage of momentum by using moving average of the gradients:
"""

optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
print(optimiser)

"""A few things to set up now before getting to the main loop:"""

# initialising step variable
step = 0

# defining the loss function
# refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
criterion = nn.CrossEntropyLoss()

# to plot losses
liveloss = PlotLosses()

# to keep the logs for loss plots
logs = {}

print('done!')

"""Now, we get to the main loop.

How much you train your model is often dependant on how much data you have, how representative of the real world your training data is and model capacity. We are going to train the network for *5,000 steps*.

Every 50 steps, we are going to evaluate the model and log the training and testing loss and accuracy. We will use [livelossplot](https://pypi.org/project/livelossplot/0.1.2/) to plot losses in the notebook but you can use a variety of packages to do the same.

"""

# outer loop - going over the steps we want to train for
while step < 5000:
  # inner loop - iterating over the batches
  for i, batch in enumerate(train_loader):

    # getting the input and the ground truth labels from the training set
    x, gt = batch
    x, gt = x.to(device), gt.to(device)

    # forward pass
    x = x.view(x.size(0), -1)
    output = model(x)

    # loss
    loss = criterion(output, gt)

    # explicitly set the gradients to zero before backpropagation
    model.zero_grad()

    # backward pass
    loss.backward()
    optimiser.step()
    step += 1

    # calculating the accuracy
    _, argmax = torch.max(output, dim=1)
    accuracy = argmax.eq(gt).float().mean() * 100

    # every 50 steps we evaluate the model on the test set
    if step % 50 == 0:

      # when we test, we don't need gradients
      with torch.no_grad():

        # beginning the loop for evaluation:
        for j, test_batch in enumerate(test_loader):

          # getting the input and the ground truth labels from the training set
          x, gt = test_batch
          x, gt = x.to(device), gt.to(device)

          # forward pass
          x = x.view(x.size(0), -1)
          output = model(x)

          # test loss
          test_loss = criterion(output, gt)

          # calculating the test accuracy
          _, argmax = torch.max(output, dim=1)
          test_accuracy = argmax.eq(gt).float().mean() * 100

        # logging train and test losses and accuracies
        logs['Loss'] = loss.item()
        logs['val_Loss'] = test_loss.item()
        logs['Accuracy'] = accuracy.item()
        logs['val_Accuracy'] = test_accuracy.item()
        liveloss.update(logs)
        liveloss.send()
        
        
"""
Copyright (c) 2023 Amir Atapour-Abarghouei, UK.

based on https://github.com/cwkx/ml-materials
License : LGPL - http://www.gnu.org/licenses/lgpl.html
""