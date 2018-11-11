# Capsule Networks

A PyTorch implementation of CapsNet based on Hinton's [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829)
using PyTorch's Visdom and Ignite. 

* [Getting Started](#getting-started)
* [Project Overview](#project-overview)
* [Running Experiments](#running-experiments)
* [Using Visdom](#using-visdom)

## Requirements 

* Python 3
* [Numpy](https://github.com/numpy/numpy)
* [Visdom](https://github.com/facebookresearch/visdom)
* [PyTorch](https://github.com/pytorch/pytorch)
* [Ignite](https://github.com/pytorch/ignite)
* [ConfigArgParse](https://github.com/bw2/ConfigArgParse  )
* [TorchVision](https://github.com/pytorch/vision)

## Getting Started

```bash
# Make sure a recent Python 3 version is installed 

# clone this repository. 
git clone git@github.com:mavanb/capsnet_pytorch.git

# install the requirements 
pip install -r requirements.txt 

# train the capsnet using the default settings 
python train_capsnet.py
```

## Project Overview

The main modules in this project are: 
* [nets.py](./nets.py): all networks or models used
* [layers.py](./layers.py): all layers used in the networks 
* [loss.py](./loss.py): margin loss
* [utils.py](./utils.py): utilities
* [train_capsnet.py](./train_capsnet.py): train an instance of the CapsNet

To handle the PyTorch training process, we use [ignite](https://github.com/pytorch/ignite). All supporting modules are 
in [ignite_features](./ignite_features). 

* [trainer.py](./ignite_features/trainer.py) contains the abstract 
Trainer class that adds all commonly used handlers and supports a train, validation and test step. The CapsuleTrainer extends this class and implement the train, valid and test functions.  
* [plot_handlers.py](./ignite_features/plot_handlers.py) handles to make standard visdom plots
* [metric.py](ignite_features/metrics.py) custom ignite metrics 
* [log_handlers.py](./ignite_features/log_handlers.py) all handlers used for logging

The default configuration file are in [default.conf](./configurations/default.conf). The data is downloaded to the [data](./data) folder.  

## Run a new experiment

To run a new experiment.

```bash
# make folder in the experiments folder
mkdir experiments/newexp

# copy the default configs 
cp configurations/default.conf  experiments/newexp/
```

Change the configurations files to the desired settings. Make sure in general.conf the experiment name points to the right experiment: 

`exp_name = newexp`

Some relevant settings: 

* Log the test accuracy

If `save_best = True` test accuracy on the best validation epoch and the model name are logged to a csv in the experiment folder. 
Change the filename using `score_file_name = best_acc`. 

* Change the architecture

The architecture of the capsule layers can be changed in the config. The default architecture is `architecture = 32,8;10,16`. 
The layers are seperated by a semi-column. Each layer constist of two numbers seperated by a comma. The number of capsule is the first number, 
the vector length the second. The primary capsule layer are arranged in a 6x6 grid, so 32 means 6x6x32 = 1152 capsules. Example of an extra layer: 
`architecture = 32,8;14,12;10,16`. 

* Change the dataset 

`dataset = mnist`. Project currently support `mnist`, `fashionmnist` and `cifar10`. The train data is split into a 
train and validation set. Change the size using `valid_size = 0.1`. 

* Debug mode

If `debug = True` the dataloader uses only one worker and only a few images are loaded into the dataset. 

## Using Visdom

[Visdom](https://github.com/facebookresearch/visdom) is used to plot and log the metrics. To use visdom, make sure that 
the general.conf file contains:

```bash
start_visdom = True
use_visdom = True
```

Or set start_visdom to False and start visdom manually: 

`python -m visdom.server -env_path ./experiments/newexp`

During training navigate to [http://localhost:8097](http://localhost:8097) to follow the training process. All visdom files are written to the env_path (generally the experiment folder) for later analysis. 