import logging

import theano
from theano.compile import Mode

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from rnn.utils import carry_hidden_state
from rnn.visualize.plot import plot


logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_states(hidden_states, updates,
                     train_stream, valid_stream,
                     args):

    # Get all the hidden_states
    filter_states = VariableFilter(theano_name_regex="hidden_state_.*")
    all_states = filter_states(hidden_states)
    all_states = sorted(all_states, key=lambda var: var.name[-1])

    # Get all the hidden_cells
    filter_cells = VariableFilter(theano_name_regex="hidden_cells_.*")
    all_cells = filter_cells(hidden_states)
    all_cells = sorted(all_cells, key=lambda var: var.name[-1])

    # Handle the theano shared variables that allow carrying the hidden state
    givens, f_updates = carry_hidden_state(updates)

    # Compile the function
    logger.info("The compilation of the function has started")
    if args.rnn_type == "lstm" and args.visualize_cells:
        compiled = theano.function(inputs=ComputationGraph(all_cells).inputs,
                                   outputs=all_cells,
                                   givens=givens, updates=f_updates,
                                   mode=Mode(optimizer='fast_compile'))
    else:
        compiled = theano.function(inputs=ComputationGraph(all_states).inputs,
                                   outputs=all_states,
                                   givens=givens, updates=f_updates,
                                   mode=Mode(optimizer='fast_compile'))

    # Plot the function
    plot("hidden_state", train_stream, compiled, args)
