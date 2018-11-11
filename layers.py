""" Layer module.

This module contains all layers used in the network module.

References:
    [1] S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic routing between capsules,” in NIPS, pp. 3859–3869, 2017.

"""

import torch
from torch import nn
from utils import squash, init_weights, get_device


class DynamicRouting(nn.Module):
    """ Dynamic routing procedure.

    Routing-by-agreement as in [1].

    Args:
        j (int): Number of parent capsules.
        n (int): Vector length of the parent capsules.
        bias_routing (bool): Add a bias parameter to the average parent predictions.
    """

    def __init__(self, j, n, bias_routing):
        super().__init__()
        self.soft_max = nn.Softmax(dim=1)
        self.j = j
        self.n = n

        # init depends on batch_size which depends on input size, declare dynamically in forward. see:
        # https://discuss.pytorch.org/t/dynamic-parameter-declaration-in-forward-function/427/2
        self.b_vec = None

        # init bias parameter
        if bias_routing:
            b_routing = nn.Parameter(torch.zeros(j, n))
            b_routing.data.fill_(0.1)
            self.bias = b_routing
        else:
            self.bias = None

        # log function that is called in the forward pass to enable analysis at end of each routing iter
        self.log_function = None

    def forward(self, u_hat, iters):
        """ Forward pass

        Args:
            u_hat (FloatTensor): Prediction vectors of the child capsules for the parent capsules. Shape: [batch_size,
                num parent caps, num child caps, len final caps]
            iters (int): Number of routing iterations.

        Returns:
            v_vec (FloatTensor): Tensor containing the squashed average predictions using the routing weights of the
                routing weight update. Shape: [batch_size, num parent capsules, len parent capsules]
        """

        b = u_hat.shape[0]  # batch_size
        i = u_hat.shape[2]  # number of parent capsules

        # init empty b_vec, on init would be better, but b and i are unknown there. Takes hardly any time this way.
        self.b_vec = torch.zeros(b, self.j, i, device=get_device(), requires_grad=False)
        b_vec = self.b_vec

        # loop over all routing iterations
        for index in range(iters):

            # softmax over j, weight of all predictions should sum to 1
            c_vec = self.soft_max(b_vec)

            # created unsquashed prediction for parents capsules by a weighted sum over the child predictions
            # in einsum: bij, bjin-> bjn
            # in matmul: bj1i, bjin = bj (1i)(in) -> bjn
            s_vec = torch.matmul(c_vec.view(b, self.j, 1, i), u_hat).squeeze()

            # add bias to s_vec
            if type(self.bias) == nn.Parameter:
                s_vec_bias = s_vec + self.bias

                # don't add a bias to capsules that have no activation add all
                # check which capsules where zero
                reset_mask = (s_vec.sum(dim=2) == 0)

                # set them back to zero again
                s_vec_bias[reset_mask, :] = 0
            else:
                s_vec_bias = s_vec

            # squash the average predictions
            v_vec = squash(s_vec_bias)

            # skip update last iter
            if index < (iters - 1):

                # compute the routing logit update
                # in einsum: "bjin, bjn-> bij", inner product over n
                # in matmul: bji1n, bj1n1 = bji (1n)(n1) = bji1
                b_vec_update = torch.matmul(u_hat.view(b, self.j, i, 1, self.n),
                                            v_vec.view(b, self.j, 1, self.n, 1)).view(b, self.j, i)

                # update b_vec
                # use x=x+1 instead of x+=1 to ensure new object creation and avoid inplace operation
                b_vec = b_vec + b_vec_update

            # call log function every routing iter for optional analysis
            if self.log_function:
                self.log_function(index, u_hat, b_vec, c_vec, v_vec, s_vec, s_vec_bias)

        return v_vec


class Conv2dPrimaryLayer(nn.Module):
    """ Compute grid of capsule by convolution layers.

    Create primary capsules as in [1]. The primary capsules are computed by:
     - first applying a conv layer with ReLU non-linearity to the input image
     - then applying a conv layer without non-linearity, reshape to capsules and apply squah non-linearity

    Args:
        in_channels (int): Number of channels of the input data/image.
        out_channels (int): Number of the capsules in the output grid.
        vec_len (int): Vector length of the primary capsules.
    """

    def __init__(self, in_channels, out_channels, vec_len):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vector_length = vec_len

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * vec_len, kernel_size=9, stride=2,
                         bias=True)
        self.conv = init_weights(conv)

    def forward(self, x):
        """ Forward pass

        Args:
            x (FloatTensor): Input image of shape [batch_size, in_channels, height_input, width_input]

        Returns:
            caps_raw (FloatTensor): Primary capsules in grid of shape
                [batch_size, out_channels, width grid, height grid, vec_len].
        """
        features = self.conv(x)
        _, _, h, w = features.shape
        caps_raw = features.contiguous().view(-1, self.out_channels, self.vector_length, h, w)  # [b, c, vec, h, w]
        caps_raw = caps_raw.permute(0, 1, 3, 4, 2)  # [b, c, h, w, vec]

        # squash on the vector dimension
        return squash(caps_raw)


class DenseCapsuleLayer(nn.Module):
    """ Dense Capsule Layer

    Dense capsule layer as in [1], but with optimized computation of the predictions if some of the child
    capsule are completely non-active.

    Args:
        i (int): Number of child capsules.
        j (int): Number of parent capsules.
        m (int): Vector length of the child capsules.
        n (int): Vector length of the parent capsules.
        stdev (float): Weight initialization transformation matrices.
    """

    def __init__(self, i, j, m, n, stdev):
        super(DenseCapsuleLayer, self).__init__()

        self.i = i
        self.j = j
        self.m = m
        self.n = n

        self.W = nn.Parameter(stdev * torch.randn(1, j, i, n, m))

    def forward(self, input):
        """ Forward pass

        Args:
            input (FloatTensor): Child capsules of the layer. Shape: [batch_size, i, n].

        Returns:
            FloatTensor: Tensor with predictions for each parent capsule of each non-zero child capsules. Shape:
                [batch_size, j, num non-zero child capsules, m].
        """

        b, i, m = input.shape
        n = self.n
        j = self.j
        assert i == self.i, "Unexpected number of childs as input"
        assert m == self.m, "Unexpected vector lenght as input"

        input = input.view(b, 1, input.shape[1], self.m, 1)

        # W: bjinm or 1jinm
        # input: b1jm1
        # matmul: bji(nm) * b1j(m1) = bjin1
        u_hat = torch.matmul(self.W, input).view(b, j, i, n)

        return u_hat
