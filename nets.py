""" Network module.

This module contains all used networks.

References:
    [1] S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic routing between capsules,” in NIPS, pp. 3859–3869, 2017.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.module import _addindent

from layers import Conv2dPrimaryLayer, DenseCapsuleLayer, DynamicRouting
from utils import one_hot, new_grid_size, init_weights


class _Net(nn.Module):
    """ Abstract class for a Network."""

    def __init__(self):
        super().__init__()
        self.epoch = 0

    @staticmethod
    def compute_predictions(logits):
        """Compute predictions by selecting the largest logit.

        Args:
            logits (FloatTensor): Logits of shape batch_size, number of capsules.

        Returns:
            FloatTensor: Tensor with indices of the largest logits of length batch_size.
        """
        _, index_max = logits.max(dim=1, keepdim=False)
        return index_max.squeeze()

    def compute_acc(self, logits, label):
        """ Compute accuracy with the logits and the labels.

        Args:
            logits (FloatTensor): Logits of shape batch_size, number of capsules.
            label (LongTensor): Corresponding labels of shape: batch_size

        Returns:
            FloatTensor: Tensor containing a single value, the average accuracy in the batch.
        """
        return sum(self.compute_predictions(logits) == label).float() / logits.size(0)

    @staticmethod
    def _num_parameters(module):
        """ Compute the number of parameters in the network. """
        return int(np.sum([np.prod(list(p.shape)) for p in module.parameters()]))

    def __repr__(self):
        """ Overwrite the repr method to get a nice representation."""
        tmpstr = self.__class__.__name__ + '(\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  ({}, {}): '.format(key, self._num_parameters(module)) + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr


class _CapsNet(_Net):
    """ Abstract CapsuleNet class. """


    @staticmethod
    def compute_logits(caps):
        """ Compute class logits from the capsule/vector length.

        Args:
            caps (FloatTensor): Final capsules of shape: batch_size, num final caps, length final caps.

        Returns:
            FloatTensor: Tensor with the logits of shape batch_size, num final caps.
        """
        # Norm is safe, to avoid nan's.
        return torch.sqrt((caps ** 2).sum(dim=-1, keepdim=False) + 1e-9)

    @staticmethod
    def compute_probs(logits):
        """ Compute the class probabilities from the logits using the softmax

        Args:
            logits: (tensor) logits of final capsules of shape [batch_size, num_classes]

        Returns: (tensor) the corresponding class probabilities of shape [batch_size, num_classes]
        """
        return F.softmax(logits, dim=1)

    def create_decoder_input(self, final_caps, labels=None):
        """ Construct decoder input based on class probs and final capsules.

        Flattens capsules to [batch_size, num_final_caps * dim_final_caps] and sets all values which do not come from
        the correct class/capsule to zero (masks). During training the labels are used to masks, during inference the
        max of the class probabilities.

        Args:
            final_caps (FloatTensor): Final capsules of shape: batch_size, num final caps, length final caps.
            labels (LongTensor, optional): Corresponding labels of shape: batch_size. Used to mask the decoder input if
                given, else the largest logit computed from the final_caps is used.

        Returns:
            FloatTensor: Flattend and masked version of the final capsules.

        """

        # get targets form the final_caps if not given
        if labels is None:
            targets = self.compute_predictions(self.compute_logits(final_caps))
        else:
            targets = labels

        # create one hot masks
        masks = one_hot(targets, final_caps.shape[1])

        # mask the capsules
        masked_caps = final_caps * masks[:, :, None]

        # flatten the masked, final capsules
        decoder_input = masked_caps.view(final_caps.shape[0], -1)

        return decoder_input


class BasicCapsNet(_CapsNet):
    """ Basic capsule network

    Basic capsule network. Architecture can be specified as arguments. Primary capsules are created as in the
    CapsNet [1]. Supports several layers and sparsity methods.

    Args:
        in_channels (int): Number of channels of the input imaga.
        routing_iters (int): Number of iterations of the routing algo.
        in_height (int): Height of the input image.
        in_width (int): Width of the input image.
        stdev_W (float):  Value of the weight initialization of the dense capsule layers.
        bias_routing (bool): Add bias to routing yes/no.
        arch (Architecture): Architecture of the capsule network.
        recon (bool): Use reconstruction yes/no.
    """

    def __init__(self, in_channels, routing_iters, in_height, in_width, stdev_W, bias_routing, arch, recon):

        # init parent CapsNet class with number of capsule in the final layer
        super().__init__()

        self.arch = arch
        self.routing_iters = routing_iters
        self.recon = recon

        prim_caps = arch.prim.caps
        prim_len = arch.prim.len

        # initial convolution
        conv_channels = prim_caps * prim_len
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_channels, kernel_size=9, stride=1, padding=0,
                          bias=True)
        self.conv1 = init_weights(conv1)
        self.relu = nn.ReLU()

        # compute primary capsules
        self.primary_caps_layer = Conv2dPrimaryLayer(in_channels=conv_channels, out_channels=prim_caps,
                                                     vec_len=prim_len)

        # grid of multiple primary caps channels is flattend, number of new channels: grid * point * channels in grid
        new_height, new_width = new_grid_size(new_grid_size((in_height, in_width), kernel_size=9), kernel_size=9,
                                              stride=2)
        in_features_dense_layer = new_height * new_width * prim_caps

        # init list for all hidden parts
        dense_layers = torch.nn.ModuleList()
        rout_layers = torch.nn.ModuleList()

        # set input of first layer to the primary layer
        in_caps = in_features_dense_layer
        in_len = arch.prim.len

        # loop over all other layers
        for h in arch.all_but_prim:

            # set capsules number and length to the current layer output
            out_caps = h.caps
            out_len = h.len

            dense_layer = DenseCapsuleLayer(j=out_caps, i=in_caps, m=in_len, n=out_len, stdev=stdev_W)

            rout_layer = DynamicRouting(j=out_caps, n=out_len, bias_routing=bias_routing)

            # add all in right order to layer list
            dense_layers.append(dense_layer)
            rout_layers.append(rout_layer)

            # capsules number and length to the next layer input
            in_caps = out_caps
            in_len = out_len

        self.dense_layers = dense_layers
        self.rout_layers = rout_layers

        if recon:
            self.decoder = CapsNetDecoder(arch.final.len, arch.final.caps, in_channels, in_height, in_width)

    def forward(self, x, t=None):
        """ Forward pass of the BasicCapsNet.

        Args:
            x (FloatTensor): Input data of shape: batch_size, in_channels, in_width, in_height.
            t (LongTensor, optional): The corresponding labels used to mask the decoder input. If not specified, the
                predicted label is used.

        Returns:
            logits (FloatTensor): Logits of shape batch_size, number of final capsules. Norm of the final_caps.
            recon (FloatTensor): Image reconstruction, same size as input.
            final_caps (FloatTensor): Final capsules. Shape: batch_size, num final capsules, length final capsules
        """

        # apply conv layer
        conv1 = self.relu(self.conv1(x))

        # compute grid of capsules
        primary_caps = self.primary_caps_layer(conv1)

        # flatten primary capsules
        b, c, w, h, m = primary_caps.shape
        primary_caps_flat = primary_caps.view(b, c * w * h, m)

        # set initial input of capsule part of the network
        caps_input = primary_caps_flat

        # loop over the capsule layers
        for dense_layer, rout_layer in zip(self.dense_layers, self.rout_layers):

            # compute for each child a parent prediction
            all_caps = dense_layer(caps_input)

            # take weighted average of parent prediction, weights determined based on correspondence of predictions
            caps_input = rout_layer(all_caps, self.routing_iters)

        # final capsule are the output of last layer
        final_caps = caps_input

        # compute the logits by taking the norm
        logits = self.compute_logits(final_caps)

        # flatten final caps and mask all but target if known
        decoder_input = self.create_decoder_input(final_caps, t)

        # create reconstruction
        if self.recon:
            recon = self.decoder(decoder_input)
        else:
            recon = None

        return logits, recon, final_caps


class CapsNetDecoder(nn.Module):
    """ Decoder network.

    As architecture of the decoder network, we used the same as for the CapsNet in [1].

    Args:
        final_caps_len (int): Length of the final capsules.
        final_caps_num (int): Number of final capsules.
        in_channels (int): Number of channels in the orginal input image.
        in_height (int): Height of the original input image.
        in_width (int): Width of the original input image.

    """

    def __init__(self, final_caps_len, final_caps_num, in_channels, in_height, in_width):
        super(CapsNetDecoder, self).__init__()

        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width

        self.reconstruct = nn.Sequential(
            init_weights(nn.Linear(final_caps_len * final_caps_num, 512)),
            nn.ReLU(inplace=True),
            init_weights(nn.Linear(512, 1024)),
            nn.ReLU(inplace=True),
            init_weights(nn.Linear(1024, in_channels * in_height * in_width)),
            nn.Sigmoid()
        )

    def forward(self, flat_final_caps):
        """ Forward pass of the decoder network.

        Args:
            flat_final_caps (FloatTensor): Flattend ans masked final capsules of shape: batch_size, final_caps_num *
                final_caps_len

        Returns:
             FloatTensor: Reconstruction of the original image with same shape.
        """
        return self.reconstruct(flat_final_caps).view(-1, self.in_channels, self.in_height, self.in_width)
