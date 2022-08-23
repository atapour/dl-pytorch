# Deep Learning Teaching Examples (PyTorch)

Deep Learning examples used for teaching within the Department of Computer Science at [Durham University](https://www.durham.ac.uk/) (UK) by [Dr. Amir Atapour-Abarghouei](http://www.atapour.co.uk/).

The material is presented as part of the "Deep Learning" lecture series at Durham University.

All material here has been tested with [PyTorch](https://pytorch.org/) 1.12 and Python 3.9.

---

## Running the Code:

- You may download each file as needed.
- You can also download the entire repository as follows:

```
git clone https://github.com/atapour/dl-pytorch
cd dl-pytorch
```
In this repository, you will find directories that contain examples that demonstrate different features of PyTorch Programming and Deep Learning in general. In the directories, you can find:

+ .py file - python code for the example
+ .ipynb file - Jupyter notebook for the example

- You can simply run each Python file by running:

```
python <example file name>.py
```

- You can run the notebooks using [Jupyter](https://jupyter.org/). 
- Note that it is recommended that you run the scripts (especially those that train a neural network) on GPU hardware. If you do not have access to a GPU locally, you can use free services like Google Colaboratory using the following steps:

### Running the Code in Google Colaboratory

 - Navigate to - [https://colab.research.google.com](https://colab.research.google.com)
 - Sign in with your Google account.

#### Using Google Colab Directly from Github
- Select File -> Upload Notebook... -> Github
- Paste the URL of the notebook on GitHub. For instance, for the notebook in this repository that covers PyTorch Tensors, you can use this URL: [https://github.com/atapour/dl-pytorch/blob/main/1.Tensors/1.PyTorch_Programming_Tensors.ipynb](https://github.com/atapour/dl-pytorch/blob/main/1.Tensors/1.PyTorch_Programming_Tensors.ipynb). You can change the URL depending on the notebook you would like to run.

#### Uploading the Notebook from the Local Copy

 - Select File -> Upload Notebook...
 - Drag and drop or browse to select the notebook you wish to use (e.g., 1.Tensors/`PyTorch_Programming_Tensors.ipynb`).


 ### Important Note

 - If a program is specifically written to use a GPU, make sure you enable the use of a GPU in [Google Colab](https://colab.research.google.com).

 - Select Runtime -> Change runtime type -> GPU

 - Alternatively, you can change the first code cell of the notebook to use a CPU to run the code by including `device = torch.device('cpu')`.

## Contents

In this repository, you can find the following examples:

#### - 0. Setup

This directory contains examples (```PyTorch_Programming_Setup.py``) that demonstrate how a simple PyTorch environment can be setup andhow visdom works.

Video: [https://youtu.be/k-VpBk81k-U](https://youtu.be/k-VpBk81k-U)

#### - 1. Tensors:

This directory contains examples (```PyTorch_Programming_Tensors.py``` and ```PyTorch_Programming_Tensors.ipynb```) that demonstrate the functionalities of Tensors in PyTorch.

Video: [https://youtu.be/enShn2dhlPo](https://youtu.be/enShn2dhlPo)

#### - 2. Datasets:

This directory contains the dataset "AckBinks: A Star Wars Dataset", which is used to demonstrate how PyTorch handles datasets. It also contains examples (```PyTorch_Programming_Datasets.py``` and ```PyTorch_Programming_Datasets.ipynb```) that show how PyTorch deals with datasets and what tools are available to process data.

Video: [https://youtu.be/UIk0MgOsa6c](https://youtu.be/UIk0MgOsa6c)

#### - 3. Backpropagation:

This directory contains examples (```PyTorch_Programming_Backpropagation.py``` and ```PyTorch_Programming_Backpropagation.ipynb```) that demonstrate how PyTorch enables backpropagation.

Video: [https://youtu.be/mLc78Vcqv-g](https://youtu.be/mLc78Vcqv-g)

#### - 4. Classifier:

This directory contains examples (```PyTorch_Programming_Classifier.py``` and ```PyTorch_Programming_Classifier.ipynb```) that provide an example of training a complete classifier using a simple neural network.

Video: [https://youtu.be/Yvvm3w3jLfg](https://youtu.be/Yvvm3w3jLfg)

---