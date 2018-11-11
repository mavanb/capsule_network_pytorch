""" Module with ignite trainers

Contains an abstract Trainer class which handles most required features of a deep learning training process. The class
uses ignite as an engine to train the model. The trainer class must be implemented by a class that implements the train,
valid and test functions. Optionally, additional custom events can be specified in the add_custom_events function.
"""

from __future__ import print_function

import time
import sys
import os
import socket
from subprocess import Popen, PIPE
import numpy as np
import torch
from torch.utils.data.sampler import SequentialSampler
from ignite.engine.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, EarlyStopping
import visdom

from data import get_train_valid_data
from ignite_features.log_handlers import SaveBestScore
from ignite_features.log_handlers import LogTrainProgressHandler, LogEpochMetricHandler
from ignite_features.metrics import ValueEpochMetric, ValueIterMetric, TimeMetric
from ignite_features.plot_handlers import VisEpochPlotter, VisIterPlotter
from utils import get_device, get_logger


class Trainer:
    """ Abstract Trainer class.

    Helper class to support a ignite training process. Call run to start training. The main tasks are:
        - init visdom
        - set seed
        - log model architecture and parameters to file or console
        - limit train / valid samples in debug mode
        - split train data into train and validation
        - load model if required
        - init train and validate ignite engines
        - sets main metrics (both iter and epoch): loss and acc
        - add default events: model saving (each epoch), early stopping, log training progress
        - calls the validate engine after each training epoch, which runs one epoch.

    When extending this class, implement the following functions:
        - _train_function: executes a training step. It takes the ignite engine, this class and the current batch as
        arguments. Should return a dict with keys:
            - 'loss': metric of this class
            - 'acc': metric of this class
            - any key that is expected by the custom events of the child class
        - _validate_function: same as _train_function, but for validate

    Optionally extend:
        - _add_custom_events: function in which additional events can be added to the training process

    Args:
        model (_Net): model/network to be trained.
        loss (_Loss): loss of the model
        optimizer (Optimizer): optimizer used in gradient update
        dataset (Dataset): dataset of torch.Dataset class
        conf (Namespace): configuration obtained using configurations.general_confs.get_conf
    """

    def __init__(self, model, loss, optimizer, data_train, data_test, conf):

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.conf = conf
        self.device = get_device()

        self._log = get_logger(__name__)

        self.vis = self._init_visdom()

        # print number of parameters in model
        num_parameters = np.sum([np.prod(list(p.shape)) for p in model.parameters()])
        self._log.info("Number of parameters model: {}".format(num_parameters))
        self._log.info("Model architecture: \n" + str(model))

        # init data sets
        kwargs = {}
        if self.device == "cuda":
            cuda_kwargs = {"pin_memory": True, "num_workers": 0}
            kwargs = {**cuda_kwargs}
        else:
            cuda_kwargs = {}
        if conf.debug:
            kwargs["train_max"] = 4
            kwargs["valid_max"] = 4
            kwargs["num_workers"] = 1
        if conf.seed:
            kwargs["seed"] = conf.seed
        self.train_loader, self.val_loader = get_train_valid_data(data_train, valid_size=conf.valid_size,
                                                                  batch_size=conf.batch_size,
                                                                  drop_last=conf.drop_last,
                                                                  shuffle=conf.shuffle, **kwargs)

        test_debug_sampler = SequentialSampler(list(range(3 * conf.batch_size))) if conf.debug else None
        self.test_loader = torch.utils.data.DataLoader(data_test, batch_size=conf.batch_size, drop_last=conf.drop_last,
                                                       sampler=test_debug_sampler, **cuda_kwargs)

        # model to cuda if device is gpu
        model.to(self.device)

        # optimize cuda
        torch.backends.cudnn.benchmark = conf.cudnn_benchmark

        # load model
        if conf.load_model:
            if os.path.isfile(conf.model_load_path):
                if torch.cuda.is_available():
                    model = torch.load(conf.model_load_path)
                else:
                    model = torch.load(conf.model_load_path, map_location=lambda storage, loc: storage)
                self._log.info(f"Succesfully loaded {conf.load_name}")
            else:
                raise FileNotFoundError(f"Could not found {conf.model_load_path}. Fix path or set load_model to False.")

        # init an ignite engine for each data set
        self.train_engine = Engine(self._train_function)
        self.valid_engine = Engine(self._valid_function)
        self.test_engine = Engine(self._test_function)

        # add train metrics
        ValueIterMetric(lambda x: x["loss"]).attach(self.train_engine, "batch_loss")  # for plot and progress log
        ValueIterMetric(lambda x: x["acc"]).attach(self.train_engine, "batch_acc")  # for plot and progress log

        # add visdom plot for the training loss
        training_loss_plot = VisIterPlotter(self.vis, "batch_loss", "Loss", "Training Batch Loss", self.conf.model_name)
        self.train_engine.add_event_handler(Events.ITERATION_COMPLETED, training_loss_plot)

        # add visdom plot for the training accuracy
        training_acc_plot = VisIterPlotter(self.vis, "batch_acc", "Acc", "Training Batch Acc", self.conf.model_name)
        self.train_engine.add_event_handler(Events.ITERATION_COMPLETED, training_acc_plot)

        # add logs handlers, requires the batch_loss and batch_acc metrics
        self.train_engine.add_event_handler(Events.ITERATION_COMPLETED, LogTrainProgressHandler())

        # add metrics
        ValueEpochMetric(lambda x: x["acc"]).attach(self.valid_engine, "acc")  # for plot and logging
        ValueEpochMetric(lambda x: x["loss"]).attach(self.valid_engine, "loss")  # for plot, logging and early stopping
        ValueEpochMetric(lambda x: x["acc"]).attach(self.test_engine, "acc")  # for plot

        # add validation acc logger
        self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                            LogEpochMetricHandler('Validation set: {:.4f}', "acc"))

        # print end of testing
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self._log.info("Done testing"))

        # saves models
        if conf.save_trained:
            save_path = f"{conf.exp_path}/{conf.trained_model_path}"
            save_handler = ModelCheckpoint(save_path, conf.model_name,
                                           score_function=lambda engine: engine.state.metrics["acc"],
                                           n_saved=conf.n_saved,
                                           require_empty=False)
            self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, save_handler, {'': model})

        # valid acc visdom plot
        acc_valid_plot = VisEpochPlotter(vis=self.vis, metric="acc", y_label="acc", title="Valid Accuracy",
                                         env_name=self.conf.model_name)
        self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, acc_valid_plot)

        # test acc visdom plot
        acc_test_plot = VisEpochPlotter(vis=self.vis, metric="acc", y_label="acc", title="Test Accuracy",
                                        env_name=self.conf.model_name)
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, acc_test_plot)

        # print ms per training example
        if self.conf.print_time:
            TimeMetric(lambda x: x["time"]).attach(self.train_engine, "time")
            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, LogEpochMetricHandler(
                'Time per example: {:.6f} ms', "time"))

        # save test acc of the best validation epoch to file
        if self.conf.save_best:

            # Add score handler for the default inference: on valid and test the same sparsity as during training
            best_score_handler = SaveBestScore(score_valid_func=lambda engine: engine.state.metrics["acc"],
                                               score_test_func=lambda engine: engine.state.metrics["acc"],
                                               start_epoch=model.epoch,
                                               max_train_epochs=self.conf.epochs,
                                               model_name=self.conf.model_name,
                                               score_file_name=self.conf.score_file_name,
                                               root_path=self.conf.exp_path)
            self.valid_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_valid)
            self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, best_score_handler.update_test)

        # add events custom events of the child class
        self._add_custom_events()

        # add early stopping, use total loss over epoch, stop if no improvement: higher score = better
        if conf.early_stop:
            early_stop_handler = EarlyStopping(patience=1,
                                               score_function=lambda engine: -engine.state.metrics["loss"],
                                               trainer=self.train_engine)
            self.valid_engine.add_event_handler(Events.COMPLETED, early_stop_handler)

        # set epoch in state of train_engine to model epoch at start to resume training for loaded model.
        # Note: new models have epoch = 0.
        @self.train_engine.on(Events.STARTED)
        def update_epoch(engine):
            engine.state.epoch = model.epoch

        # update epoch of the model, to make sure the is correct of resuming training
        @self.train_engine.on(Events.EPOCH_COMPLETED)
        def update_model_epoch(_):
            model.epoch += 1

        # makes sure eval_engine is started after train epoch, should be after all custom train_engine epoch_completed
        # events
        @self.train_engine.on(Events.EPOCH_COMPLETED)
        def call_valid(_):
            self.valid_engine.run(self.val_loader, self.train_engine.state.epoch)

        @self.train_engine.on(Events.ITERATION_COMPLETED)
        def check_nan(_):
            assert all([torch.isnan(p).nonzero().shape == torch.Size([0]) for p in model.parameters()]), \
                "Parameters contain NaNs. Occurred in this iteration."

        # makes sure test_engine is started after train epoch, should be after all custom valid_engine epoch_completed
        # events
        @self.valid_engine.on(Events.EPOCH_COMPLETED)
        def call_test(_):
            self.test_engine.run(self.test_loader, self.train_engine.state.epoch)

        # make that epoch in valid_engine and test_engine gives correct epoch (same train_engine was during run),
        # but makes sure only runs once
        @self.valid_engine.on(Events.STARTED)
        @self.test_engine.on(Events.STARTED)
        def set_train_epoch(engine):
            engine.state.epoch = self.train_engine.state.epoch - 1

        # Save the visdom environment
        @self.test_engine.on(Events.EPOCH_COMPLETED)
        def save_visdom_env(_):
            if isinstance(self.vis, visdom.Visdom):
                self.vis.save([self.conf.model_name])

    def _init_visdom(self):

        if self.conf.use_visdom:

            # start visdom if in conf
            if self.conf.start_visdom:

                # create visdom enviroment path if not exists
                if not os.path.exists(self.conf.exp_path):
                    os.makedirs(self.conf.exp_path)

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    port = 8097
                    while s.connect_ex(('localhost', port)) == 0:
                        port += 1
                        if port == 8999:
                            break

                proc = Popen([f"{sys.executable}", "-m", "visdom.server", "-env_path",
                              self.conf.exp_path, "-port", str(port), "-logging_level", "50"])
                time.sleep(1)

                vis = visdom.Visdom()

                retries = 0
                while (not vis.check_connection()) and retries < 10:
                    retries += 1
                    time.sleep(1)

                if not vis.check_connection():
                    raise RuntimeError("Could not start Visdom")

            # if use existing connection
            else:
                vis = visdom.Visdom()

                if vis.check_connection():
                    self._log.info("Use existing Visdom connection")

                # if no connection and not start
                else:
                    raise RuntimeError("Start visdom manually or set start_visdom to True")
        else:
            vis = None

        return vis


    def run(self):
        """ Start the training process. """
        self.train_engine.run(self.train_loader, max_epochs=self.conf.epochs)

    def _add_custom_events(self):
        pass

    def _train_function(self, engine, batch):
        raise NotImplementedError("Please implement abstract function _train_function.")

    def _valid_function(self, engine, batch):
        raise NotImplementedError("Please implement abstract function _valid_function.")

    def _test_function(self, engine, batch):
        raise NotImplementedError("Please implement abstract function _test_function.")
