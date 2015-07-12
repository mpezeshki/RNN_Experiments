import logging
import re

import numpy as np

import theano
from theano.compile import Mode

from blocks.graph import ComputationGraph
from rnn.datasets.dataset import conv_into_char

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    epoch_iterator = train_stream.get_epoch_iterator()
    for num in range(10):
        init_ = next(epoch_iterator)[0][
            0: args.visualize_length, 0:1]

        hidden_state = compiled(init_)

        layers = len(hidden_state)
        time = hidden_state[0].shape[0]
        ticks = tuple(conv_into_char(init_[:, 0], args.dataset))

        for d in range(layers):
            plt.subplot(layers, 1, d + 1)
            for j in range(args.state_dim):
                plt.plot(np.arange(time), hidden_state[d][:, 0, j])
            plt.xticks(range(args.visualize_length), ticks)
            plt.grid(True)
            plt.title("hidden_state_of_layer_" + str(d))
        plt.savefig(args.save_path + "/visualize_states_" + str(num) + ".png")
        logger.info("Figure \"visualize_states_" + str(num) +
                    ".png\" saved at directory: " + args.save_path)
