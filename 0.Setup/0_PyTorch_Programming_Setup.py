''' 
Video: https://youtu.be/k-VpBk81k-U

You can create an environment suitable for 'Deep Learning'. Conda provides an easy tool set.
You can download minicoda here:

https://docs.conda.io/en/latest/miniconda.html

After downloading, installing and activating the environment, you can set up all the packages you need for deep learning.

In this course, we use torch and torchvision along with a number of other packages that make our lives easier.

To run visdom, you can run:

visdom -port 1234
'''

'''
In python, you can run:
'''

import torch
import visdom
vis = visdom.Visdom(port=1234)

'''
To test out the features of visdom:
'''

vis.text('hello world!')
vis.text('nice to see you')
vis.image(torch.rand(3,256,256))

for i in range(200):
    vis.image(torch.rand(3,256,256), win='win_label', caption='my label')

vis.line(torch.rand(20))
vis.histogram(torch.randn(1000))
vis.scatter(torch.randn(10,3))
