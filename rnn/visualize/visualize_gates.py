import logging

import numpy as np

import theano

from blocks.graph import ComputationGraph

import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_gates_soft(gate_values, hidden_states, updates,
                         train_stream, valid_stream,
                         args):

    # Handle the theano shared variables for the state
    state_vars = [theano.shared(
        v[0:1, :].zeros_like().eval(), v.name + '-gen')
        for v, _ in updates]
    givens = [(v, x) for (v, _), x in zip(updates, state_vars)]
    f_updates = [(x, upd) for x, (_, upd) in zip(state_vars, updates)]

    # Compile the function
    compiled = theano.function(inputs=ComputationGraph(gate_values).inputs,
                               outputs=gate_values,
                               givens=givens, updates=f_updates,
                               mode='FAST_COMPILE')

    # Generate
    epoch_iterator = valid_stream.get_epoch_iterator()
    for i in range(10):
        init_ = next(epoch_iterator)[0][0: args.visualize_length, 0:1]

        last_output = compiled(init_)
        layers = len(last_output)
        time = last_output[0].shape[0]
        for i in range(layers):
            plt.subplot(layers, 1, i + 1)
            plt.plot(np.arange(time), last_output[i][:, 0, 0])
            plt.xticks(range(args.visualize_length), tuple(init_[:, 0]))
            plt.grid(True)
            plt.title("gate of layer " + str(i))
        plt.show()


def visualize_gates_lstm(gate_values, hidden_states, updates,
                         train_stream, valid_stream,
                         args):

    in_gates = gate_values["in_gates"]
    out_gates = gate_values["out_gates"]
    forget_gates = gate_values["forget_gates"]

    state_vars = [theano.shared(
        v[0:1, :].zeros_like().eval(), v.name + '-gen')
        for v, _ in updates]
    givens = [(v, x) for (v, _), x in zip(updates, state_vars)]
    f_updates = [(x, upd) for x, (_, upd) in zip(state_vars, updates)]

    generate_in = theano.function(inputs=ComputationGraph(in_gates).inputs,
                                  outputs=in_gates,
                                  givens=givens,
                                  updates=f_updates,
                                  mode='FAST_COMPILE')
    generate_out = theano.function(inputs=ComputationGraph(out_gates).inputs,
                                   outputs=out_gates,
                                   givens=givens,
                                   updates=f_updates,
                                   mode='FAST_COMPILE')
    generate_forget = theano.function(inputs=ComputationGraph(forget_gates).inputs,
                                      outputs=forget_gates,
                                      givens=givens,
                                      updates=f_updates,
                                      mode='FAST_COMPILE')

    # Generate
    epoch_iterator = valid_stream.get_epoch_iterator()
    for _ in range(10):
        init_ = next(epoch_iterator)[0][0: args.visualize_length, 0:1]

        last_output_in = generate_in(init_)
        last_output_out = generate_out(init_)
        last_output_forget = generate_forget(init_)
        layers = len(last_output_in)

        time = last_output_in[0].shape[0]

        for i in range(layers):

            plt.subplot(3, layers, i * 3 + 1)
            for j in range(last_output_in[i].shape[2]):
                plt.plot(np.arange(time), last_output_in[i][:, 0, j])
            plt.xticks(range(args.visualize_length), tuple(init_[:, 0]))
            plt.grid(True)
            plt.title("in_gate of layer " + str(i))

            plt.subplot(3, layers, i * 3 + 2)
            for j in range(last_output_in[i].shape[2]):
                plt.plot(np.arange(time), last_output_out[i][:, 0, j])
            plt.xticks(range(args.visualize_length), tuple(init_[:, 0]))
            plt.grid(True)
            plt.title("out_gate of layer " + str(i))

            plt.subplot(3, layers, i * 3 + 3)
            for j in range(last_output_in[i].shape[2]):
                plt.plot(np.arange(time), last_output_forget[i][:, 0, j])
            plt.xticks(range(args.visualize_length), tuple(init_[:, 0]))
            plt.grid(True)
            plt.title("forget_gate of layer " + str(i))
        plt.show()
