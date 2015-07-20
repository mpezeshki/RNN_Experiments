import logging

import numpy as np

import theano
from theano import tensor
from theano.compile import Mode

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from rnn.datasets.dataset import conv_into_char
from rnn.utils import carry_hidden_state

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_gradients(hidden_states, updates,
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

    # Get the variable on which we compute the gradients
    filter_pre_rnn = VariableFilter(theano_name_regex="pre_rnn.*")
    wrt = filter_pre_rnn(ComputationGraph(hidden_states).variables)
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

    # Comupute the gradients of states or cells
    if args.rnn_type == "lstm" and args.visualize_cells:
        states = all_cells
    else:
        states = all_states

    logger.info("The computation of the gradients has started")
    gradients = []
    for i, state in enumerate(states):
        gradients.extend(
            tensor.grad(tensor.mean(tensor.abs_(
                state[-1, 0, :])), wrt[:i + 1]))
    # -1 indicates that gradient is gradient of the last time-step.c
    logger.info("The computation of the gradients is done")

    # Handle the theano shared variables that allow carrying the hidden state
    givens, f_updates = carry_hidden_state(updates, 1)

    # Compile the function
    logger.info("The compilation of the function has started")
    compiled = theano.function(inputs=ComputationGraph(states).inputs,
                               outputs=gradients,
                               givens=givens, updates=f_updates,
                               mode=Mode(optimizer='fast_compile'))
    logger.info("The function has been compiled")

    # importing plt
    import matplotlib
    if not args.local:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Generate
    epoch_iterator = train_stream.get_epoch_iterator()
    for num in range(10):
        init_ = next(epoch_iterator)[0][
            0: args.visualize_length, 0:1]

        # [layers * len_wrt] [Time, 1, Hidden_dim]
        gradients = compiled(init_)

        if args.skip_connections:
            assert len(gradients) == (args.layers * (args.layers + 1)) / 2
        else:
            assert len(gradients) == args.layers

        time = gradients[0].shape[0]
        ticks = tuple(conv_into_char(init_[:, 0], args.dataset))

        # One row subplot for each variable wrt which we are computing
        # the gradients
        for var in range(len_wrt):
            plt.subplot(len_wrt, 1, var + 1)
            for d in range(args.layers - var):
                plt.plot(
                    np.arange(time),
                    np.mean(np.abs(gradients[d][:, 0, :]), axis=1),
                    label="layer " + str(d + var))
            plt.xticks(range(args.visualize_length), ticks)
            plt.grid(True)
            plt.yscale('log')
            axes = plt.gca()
            axes.set_ylim([5e-20, 5e-1])
            plt.title("gradients plotting w.r.t pre_rrn" + str(var))
            plt.legend()
        plt.tight_layout()
        if args.local:
            plt.show()
        else:
            plt.savefig(
                args.save_path + "/visualize_gradients_" + str(num) + ".png")
            logger.info("Figure \"visualize_gradients_" + str(num) +
                        ".png\" saved at directory: " + args.save_path)
