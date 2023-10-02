# -*- coding: utf-8 -*-
"""
# PyTorch Programming - Demonstrating Tensors
---

## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@durham.ac.uk

This notebook will provide a few examples that show the capabilities of tensors in PyTorch Programming.

Let's start by importing what we need:
"""

import torch

"""One of the most interesting things about PyTorch is that tensors are designed to behave similarly to NumPy arrays. Let's start by creating a simple rank 1 tensor:"""

x = torch.zeros(10)
print('done!')

"""and then we can inspect the tensor we have created:"""

print(x)
print(x.size())

"""PyTorch tensors support many of the operations you would expect from a NumPy array:"""

print(x+2)

print(x+3 * torch.eye(10))

"""You see that we can achieve interesting results using broadcasting. Let's create a rank 2 tensor (matrix) sampled from a Normal distribution:"""

x = torch.randn(10,5)
# torch.randn returns a tensor drawn from standard normal (mean of 0, variance of 1 - Gaussian)
print(x)
print(x.shape)

"""We can also look at a matrix drawn from the uniform distribution:"""

x = torch.rand(10,5)
# torch.randn returns a tensor drawn the uniform distribution in the interval [0,1)
print(x)
print(x.shape)

"""We can add or remove singleton dimensions to tensors using `unsqueeze` and `squeeze`.

Let's start by adding dimensions to the tensor we created earlier:
"""

print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)
print(x.unsqueeze(2).shape)

"""Note that these operations are not in-place so we haven't actually modified the `x` tensor:"""

print(x.shape)

"""However, PyTorch does actually support in-place operations. For many operations, by adding underscore, we can get the in-place version of the command - for instance:"""

x.unsqueeze_(1)
print(x.shape)

"""This, of course, could have been done through assignment as well:"""

x = x.unsqueeze(3)
print(x.shape)

"""Note that we can remove dimensions using the `squeeze` operator:"""

print(x.squeeze(3).shape)

"""Now that we have added singleton dimensions to the tensor, we can use `repeat` to repeat along the dimensions we have added:"""

x = x.repeat(1, 3, 1, 6)
print(x.shape)

"""PyTorch also supports indexing the same way NumPy does, which is very useful.

For instance, let's look at the size of the output when we index into the zeroth dimension:
"""

print(x[0].shape)

"""We can use the colon operator, `:`, to select data along specific spatial dimensions:"""

print(x[:,0].shape)

"""We can also select elements up to a point using the colon operator:"""

print(x[:3].shape)

"""or we can do it the other way around where we pick everything after a given index:"""

print(x[3:].shape)

"""We can always "`view`" the elements of our tensor in different ways. Let's look at the size of our tensor again and see how many elements it has got:"""

print(f'shape of the tensor: {x.shape}')
num_elements = torch.numel(x)
print(f'number of elements in the tensor: {num_elements}')

"""Now we can try to change our view of the tensor:"""

x.view(num_elements).shape

"""We can also do this in a different way:"""

x.view(-1).size()

"""We can actually view the tensor in a number of ways, as long as the number of elements remains the same:"""

x.view([10,90]).shape

print(x.view(x.size(2), -1).shape)

"""Let's take a look at tensor types in PyTorch.

By default, tensors are of type float and on the CPU
"""

x.dtype

"""We can easily cast tensors to other types - for instance 64-bit integer, `long`: """

x = (x*10).long()
# let us look at a piece of this as it might be too big to view the whole thing:
print(x[0])

"""Let's confirm what "device" the tensor is on:"""

x.device

"""The interesting thing is that can transfer the tensor and any subsequent operations to the GPU very easily.

Let's recreate our tensor first:


"""

x = torch.zeros(10)
print('done!')

x = x.cuda()

(x+2).device

"""GPU operations are of course much faster and are basically the engine that run deep learning.

At certain points, you might need to use CPU again, which can be done very easily:
"""

x.cpu().device

"""As you see, PyTorch offers the versatility of numpy arrays but the ability to use the GPU easily."""

"""
Copyright (c) 2023 Amir Atapour-Abarghouei, UK.

based on https://github.com/cwkx/ml-materials
License : LGPL - http://www.gnu.org/licenses/lgpl.html
"""