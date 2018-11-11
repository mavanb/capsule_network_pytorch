""" Log Handlers

This module contains all handlers that are used to log info to file or stdout.
"""

from abc import abstractmethod
import os
import logging


class LogMetricHandler:
    def __init__(self, message, metric_name, log_every=1):
        self.log_every = log_every
        self.message = message
        self.metric_name = metric_name

    def __call__(self, engine):
        assert self.metric_name in engine.state.metrics.keys(), \
            f"Engine state does not contain the metric {self.metric_name}"
        should_log, output = self.get_output(engine)
        if should_log:
            print(self.message.format(output))

    @abstractmethod
    def get_output(self, engine):
        raise NotImplementedError()


class LogIterMetricHandler(LogMetricHandler):

    def get_output(self, engine):
        return engine.state.iteration % self.log_every == 0, engine.state.metrics[self.metric_name]


class LogEpochMetricHandler(LogMetricHandler):
    def get_output(self, engine):
        return engine.state.iteration % self.log_every == 0, engine.state.metrics[self.metric_name]


class LogTrainProgressHandler:

    def __call__(self, engine):

        # -1 and +1 because counting from 1, to make sure it prints 1/X and X/X
        iteration_in_epoch = ((engine.state.iteration - 1) % len(engine.state.dataloader)) + 1

        # new line if last iteration
        end = "\n" if len(engine.state.dataloader) == iteration_in_epoch else ""

        # new line before new iteration
        start = "\n" if iteration_in_epoch == 1 else ""

        print(f"\r{start}Epoch[{engine.state.epoch}/{engine.state.max_epochs}] "
                    f"Iteration[{iteration_in_epoch}/{len(engine.state.dataloader)}] "
                    f"Acc: {engine.state.metrics['batch_acc']:.3f} "
                    f"Loss: {engine.state.metrics['batch_loss']:.3f}", end=end)


class SaveBestScore:
    """SaveBestScore handler can be used to save the best score to file (high score is considered better).

    Args:
        score_valid_func (callable): function to retrieve the valid score from the valid engine
        score_test_func (callable): function to retrieve the test score from the test engine
        start_epoch (int): epoch from
        max_train_epochs (int): number of training epochs, write best test score at max, is written to file
        model_name (str): name of the model, is written to file
        score_file_name (str): name of the file
        root_path (str, optional): path of the folder to save file
    """

    def __init__(self, score_valid_func, score_test_func, start_epoch, max_train_epochs, model_name, score_file_name,
                 root_path="./best_acc"):

        assert callable(score_valid_func), "Argument score_function should be callable."
        assert callable(score_test_func), "Argument score_function should be callable."

        self.score_valid_func = score_valid_func
        self.score_test_func = score_test_func

        self.max_train_epochs = max_train_epochs
        self.model_name = model_name

        # keep track of best valid score to determine best valid epoch
        self.best_valid_score = None

        # best valid epoch, used to retrieve best test epoch
        self.best_valid_epoch = None

        # array of each test score per epoch
        self.test_scores = {}

        self.valid_epoch = start_epoch
        self.test_epoch = start_epoch

        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())

        self.file_path = f"{root_path}/{score_file_name}.csv"

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        if not os.path.isfile(self.file_path):
            with open(self.file_path, 'w') as outf:
                outf.write(f"ModelName;EpochBest;EpochsTotal;Score\n")

    def update_valid(self, valid_engine):
        """ Should be called by the validation engine. Determines which epoch has the best score on the
        validation set."""

        # this function should be called after a validation epoch, thus update valid_epoch accordingly
        self.valid_epoch += 1

        score = self.score_valid_func(valid_engine)

        if not self.valid_epoch == valid_engine.state.epoch:
            raise IndexError("The validation engine does not have the right epoch number. Make sure that it is called "
                             "every iteration and intialized with the correct start epoch.")
        self.valid_epoch = valid_engine.state.epoch

        if self.best_valid_score is None:
            self.best_valid_score = score
            self.best_valid_epoch = self.valid_epoch
        elif score > self.best_valid_score:
            self.best_valid_score = score
            self.best_valid_epoch = self.valid_epoch

    def update_test(self, test_engine):
        """ Should be called by the test engine. Computes the score of each test epoch and saves the test score of the
        best valid epoch to file after the last epoch. """

        # this function should be called after a test epoch, thus update test_epoch accordingly
        self.test_epoch += 1

        if not self.test_epoch == test_engine.state.epoch:
            raise IndexError("The test engine does not have the right epoch number. Make sure that it is called "
                             "every iteration and intialized with the correct start epoch.")

        # retrieve score from engine
        score = self.score_test_func(test_engine)

        # append score to list
        self.test_scores[self.test_epoch] = score

        # if final epoch, save best score to file
        if test_engine.state.epoch == self.max_train_epochs:
            self._logger.info("Save best score to file")

            # get the test score of the best validation epoch, epoch count from 1
            test_of_best_valid = self.test_scores[self.best_valid_epoch]
            with open(self.file_path, 'a') as outf:
                outf.write(
                    f"{self.model_name};{self.best_valid_epoch};{self.max_train_epochs};{test_of_best_valid:0.6}\n")
