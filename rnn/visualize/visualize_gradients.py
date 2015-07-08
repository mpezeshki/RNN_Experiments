import logging

import numpy as np

import theano
from theano import tensor

from blocks.graph import ComputationGraph

import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_gradients(cost, updates,
                        train_stream, valid_stream,
                        args):

    # Time length of the visualization
    text_length = args.visualize_length

    # Boolean to know if there are several states to see
    see_several_states = (args.layers > 1 and (
        args.skip_connections or args.skip_output)) or (
        args.rnn_type in ["soft", "hard", "clockwork"])

    # Get the hidden_state
    name = "hidden_state"
    variables = ComputationGraph(cost).variables

    outputs = [var for var in variables
               if hasattr(var.tag, 'name') and
               name == var.name]

    assert len(outputs) == 1

    if see_several_states:
        h = []
        dim = args.state_dim
        for i in range(args.layers):
            h.append(
                outputs[0][:, :, dim * i: dim * (i + 1)])
    else:
        h = outputs

    cg = ComputationGraph(h)

    assert(len(cg.inputs) == 1)
    assert(cg.inputs[0].name == "features")

    wrt_gradients = [
        var for var in ComputationGraph(cost).variables if var.name == "pre_rnn"]

    gradients = []
    for state in h:
        # gradients.append(tensor.grad(tensor.mean(state[50, 0, :]), wrt_gradients[0]))
        gradients.append(
            tensor.grad(tensor.mean(tensor.abs_(state[60, 0, :])), wrt_gradients[0]))

    # Handle the theano shared variables for the state
    state_vars = [theano.shared(
        v[0:1, :].zeros_like().eval(), v.name + '-gen')
        for v, _ in updates]
    givens = [(v, x) for (v, _), x in zip(updates, state_vars)]
    f_updates = [(x, upd) for x, (_, upd) in zip(state_vars, updates)]

    # Compile the function
    compiled = theano.function(inputs=cg.inputs, outputs=gradients,
                               givens=givens, updates=f_updates)

    # Generate
    epoch_iterator = valid_stream.get_epoch_iterator()
    for i in range(10):
        init_ = next(epoch_iterator)[0][
            0: text_length, 0:1]

        gradients = compiled(init_)

        layers = len(gradients)
        time = gradients[0].shape[0]

        for i in range(layers):
            plt.subplot(layers, 1, i + 1)
            plt.plot(
                np.arange(time), np.mean(np.abs(gradients[i][:, 0, :]), axis=1))
            plt.xticks(range(text_length), tuple(init_[:, 0]))
            plt.grid(True)
            plt.yscale('log')
            plt.title("gradient_of_layer_" + str(i))
        plt.show()
