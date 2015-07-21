from collections import OrderedDict
import logging

import theano

from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from blocks.monitoring.aggregation import MonitoredQuantity
from blocks.monitoring.evaluators import (MonitoredQuantityBuffer,
                                          AggregationBuffer)
from blocks.utils import dict_subset, reraise_as

from rnn.datasets.dataset import has_indices
from rnn.utils import carry_hidden_state

logger = logging.getLogger()


class DataStreamMonitoring(SimpleExtension, MonitoringExtension):

    """Monitors Theano variables and monitored-quantities on a data stream.
    By default monitoring is done before the first and after every epoch.
    Parameters
    ----------
    variables : list of :class:`~tensor.TensorVariable` and
        :class:`MonitoredQuantity`
        The variables to monitor. The variable names are used as record
        names in the logs.
    updates : list of tuples or :class:`~collections.OrderedDict` or None
        :class:`~tensor.TensorSharedVariable` updates to be performed
        during evaluation. This parameter is only for Theano variables.
        Be careful not to update any model parameters as this is not
        intended to alter your model in any meaningful way. A typical
        use case of this option arises when the theano function used
        for evaluation contains a call to :func:`~theano.scan` which
        might have returned shared variable updates.
    data_stream : instance of :class:`.DataStream`
        The data stream to monitor on. A data epoch is requested
        each time monitoring is done.
    """
    PREFIX_SEPARATOR = '_'

    def __init__(self, variables, data_stream, mini_batch_size, dataset,
                 state_updates, updates=None, **kwargs):
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("before_first_epoch", True)
        super(DataStreamMonitoring, self).__init__(**kwargs)
        self._evaluator = DatasetEvaluator(variables, mini_batch_size,
                                           state_updates, dataset, updates)
        self.data_stream = data_stream

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("Monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("Monitoring on auxiliary data finished")


class DatasetEvaluator(object):

    """A DatasetEvaluator evaluates many Theano variables or other quantities.
    The DatasetEvaluator provides a do-it-all method, :meth:`evaluate`,
    which computes values of ``variables`` on a dataset.
    Alternatively, methods :meth:`initialize_aggregators`,
    :meth:`process_batch`, :meth:`get_aggregated_values` can be used with a
    custom loop over data.
    The values computed on subsets of the given dataset are aggregated
    using the :class:`AggregationScheme`s provided in the
    `aggregation_scheme` tags. If no tag is given, the value is **averaged
    over minibatches**. However, care is taken to ensure that variables
    which do not depend on data are not unnecessarily recomputed.
    Parameters
    ----------
    variables : list of :class:`~tensor.TensorVariable` and
        :class:`MonitoredQuantity`
        The variable names are used as record names in the logs. Hence, all
        the names must be different.
        Each variable can be tagged with an :class:`AggregationScheme` that
        specifies how the value can be computed for a data set by
        aggregating minibatches.
    updates : list of tuples or :class:`~collections.OrderedDict` or None
        :class:`~tensor.TensorSharedVariable` updates to be performed
        during evaluation. This parameter is only for Theano variables.
        Be careful not to update any model parameters as this is not
        intended to alter your model in any meaningfullway. A typical
        use case of this option arises when the theano function used
        for evaluation contains a call to:function:`~theano.scan` which
        might have returned shared variable updates.
    """

    def __init__(self, variables, mini_batch_size, state_updates,
                 dataset, updates=None):
        theano_variables = []
        monitored_quantities = []
        for variable in variables:
            if isinstance(variable, MonitoredQuantity):
                monitored_quantities.append(variable)
            else:
                theano_variables.append(variable)
        self.theano_variables = theano_variables
        self.monitored_quantities = monitored_quantities
        variable_names = [v.name for v in variables]
        if len(set(variable_names)) < len(variables):
            raise ValueError("variables should have different names")
        self.theano_buffer = AggregationBuffer(theano_variables)
        self.monitored_quantities_buffer = MonitoredQuantityBuffer(
            monitored_quantities)
        self.dataset = dataset
        self.updates = updates
        self.mini_batch_size = mini_batch_size
        self._compile(state_updates)

    def _compile(self, state_updates):
        """Compiles Theano functions.
        .. todo::
            The current compilation method does not account for updates
            attached to `ComputationGraph` elements. Compiling should
            be out-sourced to `ComputationGraph` to deal with it.
        """
        inputs = []
        outputs = []
        updates = None

        givens, f_updates = carry_hidden_state(state_updates,
                                               self.mini_batch_size,
                                               reset=has_indices(self.dataset))

        if self.theano_buffer.accumulation_updates:
            updates = OrderedDict()
            updates.update(self.theano_buffer.accumulation_updates)
            if self.updates:
                updates.update(self.updates)
            inputs += self.theano_buffer.inputs
        inputs += self.monitored_quantities_buffer.inputs
        outputs = self.monitored_quantities_buffer.requires

        if inputs != []:
            self.unique_inputs = list(set(inputs))
            updates.update(f_updates)
            self._accumulate_fun = theano.function(self.unique_inputs,
                                                   outputs,
                                                   givens=givens,
                                                   updates=updates)
        else:
            self._accumulate_fun = None

    def initialize_aggregators(self):
        self.theano_buffer.initialize_aggregators()
        self.monitored_quantities_buffer.initialize()

    def process_batch(self, batch):
        try:
            input_names = [v.name for v in self.unique_inputs]
            batch = dict_subset(batch, input_names)
        except KeyError:
            reraise_as(
                "Not all data sources required for monitoring were"
                " provided. The list of required data sources:"
                " {}.".format(input_names))
        if self._accumulate_fun is not None:
            numerical_values = self._accumulate_fun(**batch)
            self.monitored_quantities_buffer.accumulate_quantities(
                numerical_values)

    def get_aggregated_values(self):
        values = self.theano_buffer.get_aggregated_values()
        values.update(
            self.monitored_quantities_buffer.get_aggregated_values())
        return values

    def evaluate(self, data_stream):
        """Compute the variables over a data stream.
        Parameters
        ----------
        data_stream : instance of :class:`.DataStream`
            The data stream. Only the first epoch of data is used.
        Returns
        -------
        A mapping from record names to the values computed on the provided
        dataset.
        """
        self.initialize_aggregators()
        if self._accumulate_fun is not None:
            for batch in data_stream.get_epoch_iterator(as_dict=True):
                self.process_batch(batch)
        else:
            logger.debug(
                'Only data independent variables were given,'
                'will not iterate the over data!')

        return self.get_aggregated_values()
