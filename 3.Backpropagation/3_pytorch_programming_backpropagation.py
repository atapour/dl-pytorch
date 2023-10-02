# -*- coding: utf-8 -*-
"""
# PyTorch Programming - Demonstrating Backpropagation
---

## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@durham.ac.uk

This notebook will provide a few examples that show backpropagation in PyTorch.

We are going to look at how PyTorch supports backpropagation. Let's start by creating a tensor:
"""

import torch


x = torch.zeros(4, 4)
print(x)

"""In "deep learning", we would like to be able to change the values of the parameters of our network as we train it so that the output of the network gets closer and closer to the ground truth.

In this vein, we want to calculate the gradient of the loss function with respect to the parameters so we can change the parameters accordingly.

Every tensor has a [`requires_grad`](https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html), which is `false` by default. This means new pytorch tensors are little more than numpy arrays at first. Let's confirm this:
"""

print(x.requires_grad)

"""If we perform an operation on our tensor `x` and inspect its gradient, we will find that there in nothing there by default:"""

z = x + 2
print(z)

# without setting requires_grad, the tensor has no gradient:

print(f' The gradient function is {z.grad_fn}')   # expected to be empty

"""Let's start again and this time make sure `requires_grad` is set to see what we can do.

We will delete `x` to start over cleanly

"""

del x
x = torch.zeros(4,4)
x.requires_grad = True

print(x)

"""Now, we create `z` again and see what happens when `x` has the `requires_grad` set:"""

z = x + 2
print(z)

"""Under the hood, a computational graph is being created as tensors undergo operations. This graph keeps track of every operation that is applied to the data.

We can see that the `grad_fn` is keeping track of the operations:
"""

print(f'z.grad_fn: {z.grad_fn}')

"""Let's take one more step and check the gradient function again:"""

z = z * 5
print(f'z.grad_fn: {z.grad_fn}')

"""We can even walk back through the graph and see the previous operations (not that we ever need to do that in the wild):"""

print(z.grad_fn.next_functions[0][0])

"""Imagine now that `z` is the output of our neural network. We thus want to have the output be as close as possible to the ground truth. We then need to calculate a loss function, which measures the distance between the output and the ground truth, and subsequently minimise this loss.

Let's first create a hypothetical ground truth, a tensor, the same size as `z` with the value of all ones:
"""

# ground truth:
gt = torch.ones_like(z)
print(gt)

"""We can have the loss be the mean squared error between the output, `z`, and the ground truth, `gt`:"""

loss = ((z - gt)**2).mean()
print(loss)

"""This `loss` should be a single value, which tells us how good or bad the prediction of our neural network has been and we want to minimise this loss to have the output of the our network be close to the ground truth.

Let's look at the gradient function of `loss`:
"""

print(loss.grad_fn)

"""Now is the time for **backpropagation**.

The function `.backward()` starts from the `loss` variables and moves "backward" through the computational graph and computes the gradients of the `loss` with respect to all the parameters in our neural network. These gradients are then used to change the parameters in the direction of a better output.
"""

loss.backward()
print('done!')

"""If we look at the gradient of the input, `x`, we know how much to change the parameters to minimise the overall loss:"""

print(x.grad)

"""This tool set makes training neural networks very simple and efficient."""

"""
Copyright (c) 2023 Amir Atapour-Abarghouei, UK.

based on https://github.com/cwkx/ml-materials
License : LGPL - http://www.gnu.org/licenses/lgpl.html
""