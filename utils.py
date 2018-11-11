""" Utils

General functions used throughout the project.

References:
    [1] S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic routing between capsules,” in NIPS, pp. 3859–3869, 2017.
"""

import sys
import logging
import torch


def new_grid_size(grid, kernel_size, stride=1, padding=0):
    """ Calculate new images size after convoling.

    Function calculated the size of the grid after convoling an image or feature map. Used formula from:
    https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/

    Args:
        grid (tuple of ints): Tuple with 2 ints of the dimensions of the orginal grid size.
        kernel_size (int): Size of the kernel (is a square).
        stride (int, optional): Stride used.
        padding (int, optional): Padding used.
    """
    def calc(x):
        return int((x - kernel_size + 2 * padding)/stride + 1)
    return calc(grid[0]), calc(grid[1])


def squash(tensor, dim=-1):
    """ Squash function as defined in [1].

    Args:
        tensor (Tensor): Input tensor.
        dim (int, optional): Dimension on which to apply the squash function. Vector dimension. Defaults to the last.
    """
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1. + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + 1e-7)


def one_hot(labels, depth):
    """ Create one-hot encoding matrix from vector of labels/indices.

    PyTorch does not have a one-hot function like tensorflow.

    Args:
        labels (LongTensor): Tensor of labels of shape: [batch_size].
        depth (int): Output length of one hot vectors i.e. number of classes.

    Returns:
        FloatTensor: Tensor of shape [batch_size, depth] with a one-hot representation of the labels.
    # """
    return torch.eye(depth, device=get_device()).index_select(dim=0, index=labels)


def init_weights(module, weight_mean=0, weight_stddev=0.1, bias_mean=0.1):
    """ Init weights of torch.module. """
    module.weight.data.normal_(weight_mean, weight_stddev)
    module.bias.data.fill_(bias_mean)
    return module


def get_device():
    """ Get the device on which running."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name):
    """Get info logger that logs to stdout."""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
