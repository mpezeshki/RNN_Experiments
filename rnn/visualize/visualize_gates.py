import logging

import numpy as np

import matplotlib.pyplot as plt

import theano
from theano.compile import Mode

from blocks.graph import ComputationGraph
from rnn.datasets.dataset import conv_into_char
from rnn.utils import carry_hidden_state
from rnn.visualize.plot import plot

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_gates_soft(gate_values, hidden_states, updates,
                         train_stream, valid_stream,
                         args):

    # Handle the theano shared variables that allow carrying the hidden state
    givens, f_updates = carry_hidden_state(updates)

    # Compile the function
    compiled = theano.function(inputs=ComputationGraph(gate_values).inputs,
                               outputs=gate_values,
                               givens=givens, updates=f_updates,
                               mode=Mode(optimizer='fast_compile'))

    plot("gates_soft", train_stream, compiled, args)


def visualize_gates_lstm(gate_values, hidden_states, updates,
                         train_stream, valid_stream,
                         args):

    in_gates = gate_values["in_gates"]
    out_gates = gate_values["out_gates"]
    forget_gates = gate_values["forget_gates"]

    # Handle the theano shared variables that allow carrying the hidden state
    givens, f_updates = carry_hidden_state(updates)

    generate_in = theano.function(inputs=ComputationGraph(in_gates).inputs,
                                  outputs=in_gates,
                                  givens=givens,
                                  updates=f_updates,
                                  mode=Mode(optimizer='fast_compile'))
    generate_out = theano.function(inputs=ComputationGraph(out_gates).inputs,
                                   outputs=out_gates,
                                   givens=givens,
                                   updates=f_updates,
                                   mode=Mode(optimizer='fast_compile'))
    generate_forget = theano.function(inputs=ComputationGraph(forget_gates).inputs,
                                      outputs=forget_gates,
                                      givens=givens,
                                      updates=f_updates,
                                      mode=Mode(optimizer='fast_compile'))

    # Generate
    epoch_iterator = valid_stream.get_epoch_iterator()
    for num in range(10):
        init_ = next(epoch_iterator)[0][0: args.visualize_length, 0:1]

        last_output_in = generate_in(init_)
        last_output_out = generate_out(init_)
        last_output_forget = generate_forget(init_)
        layers = len(last_output_in)

        time = last_output_in[0].shape[0]
        ticks = tuple(conv_into_char(init_[:, 0], args.dataset))

        for i in range(layers):

            plt.subplot(3, layers, 1 + i)
            plt.plot(np.arange(time), np.mean(
                np.abs(last_output_in[i][:, 0, :]), axis=1))
            plt.xticks(range(args.visualize_length), ticks)
            plt.grid(True)
            plt.title("in_gate of layer " + str(i))

            plt.subplot(3, layers, layers + 1 + i)
            plt.plot(np.arange(time), np.mean(
                np.abs(last_output_out[i][:, 0, :]), axis=1))
            plt.xticks(range(args.visualize_length), ticks)
            plt.grid(True)
            plt.title("out_gate of layer " + str(i))

            plt.subplot(3, layers, 2 * layers + 1 + i)
            plt.plot(np.arange(time), np.mean(
                np.abs(last_output_forget[i][:, 0, :]), axis=1))
            plt.xticks(range(args.visualize_length), ticks)
            plt.grid(True)
            plt.title("forget_gate of layer " + str(i))
        plt.tight_layout()
        if args.local:
            plt.show()
        else:
            plt.savefig(
                args.save_path + "/visualize_gates_" + str(num) + ".png")
            logger.info("Figure \"visualize_gates_" + str(num) +
                        ".png\" saved at directory: " + args.save_path)
