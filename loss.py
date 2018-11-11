""" Loss module.

References:
    [1] S. Sabour, N. Frosst, and G. E. Hinton, “Dynamic routing between capsules,” in NIPS, pp. 3859–3869, 2017.

"""
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from utils import one_hot


class CapsuleLoss(_Loss):
    """ Margin Loss

    Margin Loss as defined in [1].

    Args:
        m_plus (float): m+ in the margin loss.
        m_min (float): m- in the margin loss.
        alpha (float): the scalar that controls the contribution of the reconstruction loss.
        include_recon (bool): Use the reconstruction loss.
    """

    def __init__(self, m_plus, m_min, alpha, include_recon):
        super(CapsuleLoss, self).__init__()

        self.m_plus = m_plus
        self.m_min = m_min
        self.alpha = alpha
        self.include_recon = include_recon

        # init mean square error loss
        self.recon_loss = nn.MSELoss(reduction="none")

    def forward(self, images, labels, logits, recon):
        """ Forward pass.

        Args:
            images (FloatTensor): Orginal images. Shape: [batch, channel, height, width].
            labels (LongTensor): Class labels. Shape: [batch]
            logits (FloatTensor): Class logits. Length of the final capsules. Shape: [batch, classes]
            recon (FloatTensor): Reconstructed image. Same shape as images.

        Returns:
            total_loss (FloatTensor): Sum of all losses. Single value.
            margin_loss (FloatTensor): Margin loss defined in [1]. Single value.
            recon_loss (FloatTensor): MSE loss of the reconstructed image. None if not included. Single value.
        """

        num_classes = logits.shape[1]
        labels_one_hot = one_hot(labels, num_classes)

        # the factor 0.5 in front of both terms is not in the paper, but used in the source code
        present_loss = 0.5 * F.relu(self.m_plus - logits, inplace=True) ** 2
        absent_loss = 0.5 * F.relu(logits - self.m_min, inplace=True) ** 2

        # the factor 0.5 is the downweight mentioned in the Margin loss in [1]
        margin_loss = labels_one_hot * present_loss + 0.5 * (1. - labels_one_hot) * absent_loss
        margin_loss_per_sample = margin_loss.sum(dim=1)
        margin_loss = margin_loss_per_sample.mean()

        if self.include_recon:
            # sum over all image dimensions
            recon_loss = self.recon_loss(recon, images).sum(dim=-1).sum(dim=-1).sum(dim=-1)
            assert len(recon_loss.shape) == 1, "Only batch dimension should be left after in recon loss."

            # average of sum over batch dimension
            recon_loss = recon_loss.mean()

        else:
            recon_loss = None

        # scale the recon
        total_loss = margin_loss
        if self.include_recon:
            total_loss = total_loss + self.alpha * recon_loss

        return total_loss, margin_loss, recon_loss
