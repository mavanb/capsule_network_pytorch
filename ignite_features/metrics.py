""" Metrics

This module contains all implementations of the ignite Metric class.
"""

from __future__ import division

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.engine import Events


class IterMetric(Metric):
    """ Abstract class of iteration Metric

    Metric that is computed and reset at the end of every iteration instead after each epoch.
    """

    def attach(self, engine, name):
        engine.add_event_handler(Events.ITERATION_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)


class ValueEpochMetric(Metric):
    """ Metric that computes single value every epoch.

    Calculates the average of some value that must be average of the number of batches per epoch.
    """

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        self._sum += output
        self._num_examples += 1.0

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'ValueMetric must have received at least one batch before it can be computed')
        return self._sum / self._num_examples


class ValueIterMetric(ValueEpochMetric, IterMetric):
    """ ValueMetric with is computed and reset at every iteration instead of epoch."""
    pass


class TimeMetric(Metric):
    """ Metric that calculated the average time computation per sample over an epoch."""

    def reset(self):
        self._avg_diff = 0.0
        self._prev_time = 0.0
        self._num_examples = 0

    def update(self, output):
        # convert time to ms
        new_time = output[0] * 1000

        if self._prev_time:
            batch_size = output[1]
            new_diff = (new_time - self._prev_time)
            total = self._num_examples + batch_size
            # _avg_diff gives time per sample. Thus, to update we compute the weighted average:
            # avg_diff * (num_examples / total) + avg_new_diff * (batch_size / total)
            # avg_new_diff * batch_size = (new_diff / batch_size) * batch_size = new_diff
            self._avg_diff = (self._avg_diff * self._num_examples + new_diff) / total
            self._num_examples += batch_size
        self._prev_time = new_time

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'TimeMetric must have received at least one batch before it can be computed')
        return self._avg_diff


