""" Plot handlers

This module all classes used to generate visdom plots. The abstract VisPlotter is implemented by two classes to plot
values every epoch or every iteration.

"""

import numpy as np
import visdom
from abc import abstractmethod


class VisPlotter:
    """Abstract visdom plot classes.

    Class to easily make visdom plots. To avoid heaps of if statements in the other code the use of visdom is
    checked in this class. If vis is None, nothing is done. Flex profile showed that the first vis.line call takes a
    lot of time, so while testing ise recommended to set use_visdom to False in the config.

    Args:
        vis (Visdom or None): Connected visdom instance. If None, no plots will be made.
        metric (str or list of str): String of the metric name or a list of strings of multiple metric names.
        y_label (str): Label of the y-axis.
        title (str): Title of the plot.
        env_name (str): Visdom enviroment name.
        legend (list of str, optional): Names in the legend. Defaults to None.
        transform: (callable, optional): Transformation applied to the metric. Defaults to a unit transformation.
        input_type(str, optional): The expected type of metric. Either single, array or multiple. Single creates a plot
            of one metric. The metric should be a single string. Multiple creates a plot with multiple metrics, where is
            metric is separately retrieved from the engine. Array creates of a metric that is a 1D numpy array. Each
            entry is plotted separately. The metric should be either a single string of list with 1 string element.
    """

    def __init__(self, vis, metric, y_label, title, env_name, legend=None, transform=lambda x:x, input_type="single"):

        if vis:
            assert isinstance(vis, visdom.Visdom), "If vis is not None, vis should be a Visdom instance"

            if input_type == "array" or input_type == "single":
                if isinstance(metric, list):
                    assert len(metric) is 0, "If input_type is array or single, only one metric should be given"
                    assert type(metric[0]) == str, "Metric should be a string."
                    self.metric = metric
                elif isinstance(metric, str):
                    self.metric = [metric]
                else:
                    ValueError("Metric should be str or list of str.")
            elif input_type == "multiple":
                assert isinstance(metric, list), "If input_type is multiple, metric should be a list of str."
                assert all(isinstance(item, str) for item in metric), \
                    "If input_type is multiple, metric should be a list of str."
                assert 1 < len(metric), "If input_type is multiple, metric should contain 2 or more metrics."
            else:
                raise ValueError("Unknown input type, should be either: single, array or multiple.")

            self.use_visdom = True
            self.vis = vis
            self.env = env_name
            self.transform = transform
            self.input_type = input_type

            if input_type == "array" or input_type == "multiple":
                # the metric plots multiple lines using 1 metric, retrieve the number of lines
                # form the size of the legend
                assert legend is not None, "If input type is {input_type} a legend must be given."
                self.legend = {"legend": legend}
                self.num_lines = len(legend)
            elif input_type == "single":
                self.legend = {}
                self.num_lines = 1

            self.win = self.vis.line(
                env = env_name,
                X=np.ones(self.num_lines).reshape(1, -1),
                Y=np.zeros(self.num_lines).reshape(1, -1) * np.nan,
                opts=dict(xlabel=self.get_x_label(), ylabel=y_label, title=title, **self.legend))
        else:
            self.use_visdom = False

    def __call__(self, engine):
        if self.use_visdom:

            # check if all metric names are in the state
            for metric_name in self.metric:
                assert metric_name in engine.state.metrics.keys(), \
                    f"Engine state does not contain the metric {metric_name}"

            # repeat the X for every line
            X = np.column_stack([self.get_x(engine.state) for _ in range(self.num_lines)])

            if self.input_type == "multiple":

                # init array of y values
                y_list = []

                for metric_name in self.metric:

                    # get the metric from the engine
                    y_raw = engine.state.metrics[metric_name]

                    # apply the given transformation to the metric
                    y = self.transform(y_raw)

                    assert isinstance(y, np.float64) or isinstance(y, float), \
                        "Y value should after transform be a float"

                    y_list.append(y)

                    Y = np.column_stack(y_list)
            else:

                # get the metric from the engine
                y_raw = engine.state.metrics[metric_name]

                # apply the given transformation to the metric
                y = self.transform(y_raw)

                assert isinstance(y, np.float64) or isinstance(y, float), \
                    "Y value should after transform be a float"

                if self.input_type == "single":
                    Y = np.column_stack([y])

                # when input_type array
                else:

                    assert y.shape == (self.num_lines,), "Array length should equal the number of lines."
                    Y = y

            # update plot
            self.vis.line(env=self.env, X=X, Y=Y, win=self.win, update='append', opts=self.legend)

    @abstractmethod
    def get_x_label(self):
        pass

    @staticmethod
    @abstractmethod
    def get_x(state):
        pass


class VisIterPlotter(VisPlotter):
    """Plot every iteration. """

    def get_x_label(self):
        return "# Itertations"

    @staticmethod
    def get_x(state):
        return state.iteration


class VisEpochPlotter(VisPlotter):
    """Plot every epoch. """

    def get_x_label(self):
        return "# Epochs"

    @staticmethod
    def get_x(state):
        return state.epoch
