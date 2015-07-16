import logging
import re

import numpy as np

import matplotlib.pyplot as plt

import theano
from theano import tensor
from theano.compile import Mode

from blocks.graph import ComputationGraph
from rnn.datasets.dataset import conv_into_char

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_presoft(cost, hidden_states, updates,
                      train_stream, valid_stream,
                      args):

    variables = ComputationGraph(cost).variables
    presoft = [var for var in variables if var.name == "presoft"]
    presoft = presoft[0]

    # Get all the hidden_states
    all_states = [
        var for var in hidden_states if re.match("hidden_state_.*", var.name)]
    all_states = sorted(all_states, key=lambda var: var.name[-1])

    # Assertion part
    assert len(all_states) == args.layers

    logger.info("The computation of the gradients has started")
    gradients = []

    for i in range(args.visualize_length - args.context):
        gradients.extend(
            tensor.grad(tensor.mean(tensor.abs_(presoft[i, 0, :])), all_states))
    logger.info("The computation of the gradients is done")

    # Handle the theano shared variables for the state
    state_vars = [theano.shared(
        v[0:1, :].zeros_like().eval(), v.name + '-gen')
        for v, _ in updates]
    givens = [(v, x) for (v, _), x in zip(updates, state_vars)]
    f_updates = [(x, upd) for x, (_, upd) in zip(state_vars, updates)]

    # Compile the function
    logger.info("The compilation of the function has started")
    compiled = theano.function(inputs=ComputationGraph(presoft).inputs,
                               outputs=gradients,
                               givens=givens, updates=f_updates,
                               mode=Mode(optimizer='fast_compile'))
    logger.info("The function has been compiled")

    # Generate
    epoch_iterator = train_stream.get_epoch_iterator()
    for num in range(10):
        init_ = next(epoch_iterator)[0][
            0: args.visualize_length, 0:1]

        hidden_state = compiled(init_)

        value_of_layer = {}
        for d in range(args.layers):
            value_of_layer[d] = 0

        for i in range(len(hidden_state) / args.layers):
            for d in range(args.layers):
                value_of_layer[d] += hidden_state[d + i * args.layers]

        time = hidden_state[0].shape[0]
        ticks = tuple(conv_into_char(init_[:, 0], args.dataset))

        for d in range(args.layers):
            plt.plot(
                np.arange(time),
                np.mean(np.abs(value_of_layer[d][:, 0, :]), axis=1),
                label="Layer " + str(d))
        plt.xticks(range(args.visualize_length), ticks)
        plt.grid(True)
        plt.title("hidden_state_of_layer_" + str(d))
        plt.legend()
        plt.tight_layout()
        if args.local:
            plt.show()
        else:
            plt.savefig(
                args.save_path + "/visualize_presoft_" + str(num) + ".png")
            logger.info("Figure \"visualize_presoft_" + str(num) +
                        ".png\" saved at directory: " + args.save_path)
