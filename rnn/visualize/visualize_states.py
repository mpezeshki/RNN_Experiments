import logging
import re

import numpy as np

import matplotlib.pyplot as plt

import theano
from theano.compile import Mode

from blocks.graph import ComputationGraph
from rnn.visualize.plot import plot


logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_states(hidden_states, updates,
                     train_stream, valid_stream,
                     args):

    # Get all the hidden_states
    all_states = [
        var for var in hidden_states if re.match("hidden_state_.*", var.name)]
    all_states = sorted(all_states, key=lambda var: var.name[-1])

    # Get all the hidden_cells
    all_cells = [var for var in hidden_states if re.match(
        "hidden_cell_.*", var.name)]
    all_cells = sorted(all_cells, key=lambda var: var.name[-1])

    # Handle the theano shared variables for the state
    state_vars = [theano.shared(
        v[0:1, :].zeros_like().eval(), v.name + '-gen')
        for v, _ in updates]
    givens = [(v, x) for (v, _), x in zip(updates, state_vars)]
    f_updates = [(x, upd) for x, (_, upd) in zip(state_vars, updates)]

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
