import logging
import re

import numpy as np

import theano
from theano import tensor

from blocks.graph import ComputationGraph

import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_gradients(cost, hidden_states, updates,
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

    # Get the variable on which we compute the gradients
    variables = ComputationGraph(cost).variables
    wrt = [
        var for var in variables if
        (var.name is not None) and (re.match("pre_rnn.*", var.name))]
    wrt = sorted(wrt, key=lambda var: var.name[-1])
    len_wrt = len(wrt)

    # We have wrt = [pre_rnn] or [pre_rnn_0, pre_rnn_1, ...]

    # Assertion part
    assert len(all_states) == args.layers
    assert len(all_cells) == (args.layers * (args.rnn_type == "lstm"))
    if args.skip_connections:
        assert len_wrt == args.layers
    else:
        assert len_wrt == 1

    logger.info("The computation of the gradients has started")
    gradients = []

    # This is for cells
    if args.rnn_type == "lstm" and args.visualize_cells:
        for i, state in enumerate(all_cells):
            gradients.extend(
                tensor.grad(tensor.mean(tensor.abs_(state[60, 0, :])), wrt[:i + 1]))
        logger.info("The computation of the gradients is done")

        # Handle the theano shared variables for the state
        state_vars = [theano.shared(
            v[0:1, :].zeros_like().eval(), v.name + '-gen')
            for v, _ in updates]
        givens = [(v, x) for (v, _), x in zip(updates, state_vars)]
        f_updates = [(x, upd) for x, (_, upd) in zip(state_vars, updates)]

        # Compile the function
        logger.info("The compilation of the function has started")
        compiled = theano.function(inputs=ComputationGraph(all_cells).inputs,
                                   outputs=gradients,
                                   givens=givens, updates=f_updates,
                                   mode='FAST_COMPILE')
        logger.info("The function has been compiled")

    # This is for state
    else:
        for i, state in enumerate(all_states):
            gradients.extend(
                tensor.grad(tensor.mean(tensor.abs_(state[60, 0, :])), wrt[:i + 1]))
        logger.info("The computation of the gradients is done")

        # Handle the theano shared variables for the state
        state_vars = [theano.shared(
            v[0:1, :].zeros_like().eval(), v.name + '-gen')
            for v, _ in updates]
        givens = [(v, x) for (v, _), x in zip(updates, state_vars)]
        f_updates = [(x, upd) for x, (_, upd) in zip(state_vars, updates)]

        # Compile the function
        logger.info("The compilation of the function has started")
        compiled = theano.function(inputs=ComputationGraph(all_states).inputs,
                                   outputs=gradients,
                                   givens=givens, updates=f_updates,
                                   mode='FAST_COMPILE')
        logger.info("The function has been compiled")

    # Generate
    epoch_iterator = train_stream.get_epoch_iterator()
    for _ in range(10):
        init_ = next(epoch_iterator)[0][
            0: args.visualize_length, 0:1]

        # [layers * len_wrt] [Time, 1, Hidden_dim]
        gradients = compiled(init_)

        if args.skip_connections:
            assert len(gradients) == (args.layers * (args.layers + 1)) / 2
        else:
            assert len(gradients) == args.layers

        time = gradients[0].shape[0]

        # One row subplot for each variable wrt which we are computing
        # the gradients
        for var in range(len_wrt):
            plt.subplot(len_wrt, 1, var + 1)
            for d in range(args.layers - var):
                plt.plot(
                    np.arange(time),
                    np.mean(np.abs(gradients[d][:, 0, :]), axis=1),
                    label="layer " + str(d + var))
            plt.xticks(range(args.visualize_length), tuple(init_[:, 0]))
            plt.grid(True)
            plt.yscale('log')
            axes = plt.gca()
            axes.set_ylim([5e-20, 5e-1])
            plt.title("gradients plotting w.r.t pre_rrn" + str(var))
            plt.legend()
        plt.show()
