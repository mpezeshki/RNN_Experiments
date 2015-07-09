import logging

import numpy as np

import theano

from blocks.graph import ComputationGraph

import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_states(cost, updates,
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

    # Handle the theano shared variables for the state
    state_vars = [theano.shared(
        v[0:1, :].zeros_like().eval(), v.name + '-gen')
        for v, _ in updates]
    givens = [(v, x) for (v, _), x in zip(updates, state_vars)]
    f_updates = [(x, upd) for x, (_, upd) in zip(state_vars, updates)]

    # Compile the function
    compiled = theano.function(inputs=cg.inputs, outputs=h,
                               givens=givens, updates=f_updates,
                               mode='FAST_COMPILE')

    epoch_iterator = valid_stream.get_epoch_iterator()
    for _ in range(10):
        init_ = next(epoch_iterator)[0][
            0: text_length, 0:1]

        hidden_state = compiled(init_)

        layers = len(hidden_state)
        time = hidden_state[0].shape[0]

        for i in range(layers):
            plt.subplot(layers, 1, i + 1)
            for j in range(args.state_dim):
                plt.plot(np.arange(time), hidden_state[i][:, 0, j])
            plt.xticks(range(text_length), tuple(init_[:, 0]))
            plt.grid(True)
            plt.title("hidden_state_of_layer_" + str(i))
        plt.show()
