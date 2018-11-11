"""Train basic capsule network

This module is used to train the basic capsule network.

Run using:
        $ python train_capsnet.py

To change the configuration either:
    - change default.conf
    - make a new config file and pass this as argument, example:
         $ python train_capsnet.py --conf experiments/some_new_exp/some_new.conf
    - pass as command line arguments, example:
        $ python train_capsnet.py --model_name your_model_name --routing_iters 2
"""

from __future__ import print_function

import time
import numpy as np
import torch
from torchvision import transforms

from configurations import get_conf
from data import get_dataset
from ignite_features.trainer import Trainer
from nets import BasicCapsNet
from loss import CapsuleLoss
from utils import get_logger


class CapsuleTrainer(Trainer):
    """ Trainer of a capsule network

    This class extends a the Trainer class, which adds all ignite handles that are used by most training processes.
    """

    def _train_function(self, engine, batch):

        self.model.train()
        self.optimizer.zero_grad()

        data = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        logits, recon, _ = self.model(data, labels)

        total_loss, _, _ = self.loss(data, labels, logits, recon)

        acc = self.model.compute_acc(logits, labels)

        total_loss.backward()
        self.optimizer.step()

        return {"loss": total_loss.item(), "time": (time.time(), data.shape[0]), "acc": acc.item()}

    def _valid_function(self, engine, batch):

        self.model.eval()

        with torch.no_grad():
            data = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            logits, recon, _ = self.model(data)
            loss, _, _ = self.loss(data, labels, logits, recon)

            acc = self.model.compute_acc(logits, labels).item()

        return {"loss": loss.item(), "acc": acc, "epoch": self.model.epoch}

    def _test_function(self, engine, batch):

        self.model.eval()

        with torch.no_grad():
            data = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            logits, _, _, = self.model(data)

            acc = self.model.compute_acc(logits, labels).item()

        return {"acc": acc}


def main():

    # get general config
    conf, parser = get_conf()

    # get logger and log config
    log = get_logger(__name__)
    log.info(parser.format_values())

    # seed must be set before any stochastic operation in torch or numpy
    if conf.seed:
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)

    # get data set
    transform = transforms.ToTensor()
    data_train, data_test, data_shape, label_shape = get_dataset(conf.dataset, transform=transform)

    assert conf.architecture.final.caps == label_shape, "Number of final capsule should match the number of labels."

    # init basic capsnet
    model = BasicCapsNet(in_channels=data_shape[0], routing_iters=conf.routing_iters, in_height=data_shape[1],
                         in_width=data_shape[2], stdev_W=conf.stdev_W, bias_routing=conf.bias_routing,
                         arch=conf.architecture, recon=conf.use_recon)

    # init capsule loss
    capsule_loss = CapsuleLoss(conf.m_plus, conf.m_min, conf.alpha, include_recon=conf.use_recon)

    # init adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)

    # init Trainer that supports the ignite training processs
    trainer = CapsuleTrainer(model, capsule_loss, optimizer, data_train, data_test, conf)

    # start trainer
    trainer.run()


if __name__ == "__main__":
    main()
