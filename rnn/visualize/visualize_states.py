import logging
import re

import numpy as np

import theano

from blocks.graph import ComputationGraph

import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_states(cost, hidden_states, updates,
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
                                   mode='FAST_COMPILE')
    else:
        compiled = theano.function(inputs=ComputationGraph(all_states).inputs,
                                   outputs=all_states,
                                   givens=givens, updates=f_updates,
                                   mode='FAST_COMPILE')

    epoch_iterator = train_stream.get_epoch_iterator()
    for _ in range(10):
        init_ = next(epoch_iterator)[0][
            0: args.visualize_length, 0:1]

        hidden_state = compiled(init_)

        layers = len(hidden_state)
        time = hidden_state[0].shape[0]

        for d in range(layers):
            plt.subplot(layers, 1, d + 1)
            for j in range(args.state_dim):
                plt.plot(np.arange(time), hidden_state[d][:, 0, j])
            plt.xticks(range(args.visualize_length), tuple(init_[:, 0]))
            plt.grid(True)
            plt.title("hidden_state_of_layer_" + str(d))
        plt.show()
