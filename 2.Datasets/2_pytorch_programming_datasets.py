# -*- coding: utf-8 -*-
"""
# PyTorch Programming - Demonstrating Datasets
---

## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@durham.ac.uk

This notebook will provide a few examples that show how PyTorch deals with datasets.

Let's start by importing what we need. [Torchvision](https://pytorch.org/vision/stable/index.html) is a very helpful that helps us deal with **vision** data a lot easier. There is also [Torchtext](https://pytorch.org/text/stable/index.html) and [Torchaudio](https://pytorch.org/audio/stable/index.html), which can help you with all kinds of projects.
"""

import torch
import torchvision

"""In order to see how PyTorch can deal with datasets, we need a sample dataset to work with. [Torchvision](https://pytorch.org/vision/stable/index.html) offers a variety of built-in datasets itself that can easily work out of the box (https://pytorch.org/vision/stable/datasets.html) but here, we will be working with our own dataset:

**[AckBinks: A Star Wars Dataset](https://github.com/atapour/dl-pytorch/tree/main/2.Datasets/AckBinks)**

First, let's download the dataset, which is available on the GitHub repo that contains this example:
"""

!wget -q -O AckBinks.zip https://github.com/atapour/dl-pytorch/blob/main/2.Datasets/AckBinks/AckBinks.zip?raw=true
!unzip -q AckBinks.zip
!rm AckBinks.zip
print('done!')

"""Before we get to loading the data, we need to transform the data to make them suitable for PyTorch dataloaders. To do this, we will use torchvision [transforms](https://pytorch.org/vision/stable/transforms.html).

In the following, we transform the training data to be all the same resolution (224x224), we make it so they are randomly flipped on their horizontal axis (as a form of data augmentation), we convert them to PyTorch Tensors and finally normalise their values so they are zero mean unit variance:
"""

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print('created transforms for training set!')

"""We shall do the same for our test data, though we need to be careful that we want the transforms applied to our test data to be as close as possible to real-world data:"""

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print('created transforms for testing set!')

"""Now that our transforms are ready, we can create a [Dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset) object for the test and training sets. Here, we use [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) from Torchvision to create our dataset. Note that this does not actually loading the dataset. The Dataset class basically creates a data structure telling us where the images actually are:"""

train_dataset = torchvision.datasets.ImageFolder('AckBinks/train', train_transform)
test_dataset = torchvision.datasets.ImageFolder('AckBinks/test', test_transform)

print(f"There are {len(train_dataset)} images in the training set!")
print(f"There are {len(test_dataset)} images in the test set!")

"""We can have a look at the class names in our dataset:

"""

class_names = train_dataset.classes
print(*class_names, sep = ", ")

"""The [Dataloader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) is responsible for actually loading the data. Note that since datasets can be massive, they are not meant to be loaded into memory all at once, which is why they are often loaded in "*batch*"es.

In the following, we are going to load our data in batches of 4. We will shuffle the data during training and will disable the `shuffle` flag during test time.

Note that `num_workers` refers to the number of threads, which can helps you process things faster if you have multiple processing cores available to you.
"""

train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=4, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_dataset, 
    batch_size=4, shuffle=False, num_workers=2)

print('done!')

"""Let us now explore the data we have loaded. We can get the first batch from the `train_loader` iterable and look at the Tensor:"""

x, y = next(iter(train_loader))

print(f"Size of tensor: {x.shape}")
print(f"Size of label: {y.shape}")

"""We can also try to visualise these images. We can do this using libraries like [OpenCV](https://opencv.org/), [Matplotlib](https://matplotlib.org/) or [Visdom](https://ai.facebook.com/tools/visdom/), but let's take this opportunity to take a look at [Weights and Biases](https://wandb.ai), which is becoming very popular and is commonly used in industry.

If you haven’t already, you will need to create a new account in order to be able to use Weights and Biases. it is free to use unless you are a massive corporation.

You can visit [their website](https://wandb.ai/site) and sign up. It can then be simply installed using `pip` or `conda`.
"""

!pip install wandb -Uq

"""You will then need to authenticate yourself via the `wandb login` command. You will be prompted to copy paste an authorization key in order to continue."""

import wandb

wandb.login()

"""We need to initialise a project:"""

wandb.init(project="super-duper-demo")

"""Weights and Biases has many functionalities including experiment tracking, versioning, hyperparameter optimisation and others, but for now, we will use it as a simple visualisation tool.

In the following, we create a table from our data where the columns are our images and their corresponding labels. We can then inspect the output in our dashboard. 
"""

x, y = next(iter(train_loader))

columns = ['Image', 'Label']
data = []

for i, img in enumerate(x, 0):
  data.append([wandb.Image(img), class_names[y[i].item()]])

table = wandb.Table(data=data, columns=columns)
wandb.log({"AckBink Images": table})

"""We can also try doing the same thing like a champ, without any fancy packages.

Torchvision can help us create a grid of our images, which we can then display using Matplotlib:
"""

import matplotlib.pyplot as plt

x, y = next(iter(train_loader))

grid = torchvision.utils.make_grid(x)
grid = (grid-grid.min())/(grid.max()-grid.min())

plt.imshow(grid.permute(1, 2, 0))

"""Torchvision also has numerous datasets, often used for research and benchmarking, built into it.

[https://pytorch.org/vision/stable/datasets.html](https://pytorch.org/vision/stable/datasets.html)


These built-in datasets are very easy to load and work with. For example, let's look at the seminal MNIST dataset:
"""

# MNIST example
mnist_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor()
    ])),
shuffle=True, batch_size=64, drop_last=True)
print('MINST has been loaded!')

"""Let's have a look at some of the images in the MNIST dataset:"""

x, y = next(iter(mnist_loader))

grid = torchvision.utils.make_grid(x)
grid = (grid-grid.min())/(grid.max()-grid.min())

print(f'Size of image is {x.size()}')
plt.imshow(grid.permute(1, 2, 0))

"""There are various dataset processing tools available in PyTorch, which can help you load in and pre-process your data easily in efficient ways.

https://pytorch.org/vision/stable/index.html
"""

"""
Copyright (c) 2023 Amir Atapour-Abarghouei, UK.

based on https://github.com/cwkx/ml-materials
License : LGPL - http://www.gnu.org/licenses/lgpl.html
""