import numpy as np

from blocks.model import Model
from blocks.serialization import load_parameter_values

from rnn.datasets.dataset import get_output_size
from rnn.visualize.visualize_gates import (
    visualize_gates_soft, visualize_gates_lstm)
from rnn.visualize.visualize_states import visualize_states
from rnn.visualize.visualize_gradients import visualize_gradients
# from rnn.visualize.visualize_gradients import visualize_jacobian
from rnn.visualize.visualize_presoft import visualize_presoft
from rnn.visualize.visualize_matrices import visualize_matrices
from rnn.visualize.visualize_singular_values import visualize_singular_values
from rnn.visualize.visualize_gradients_flow_pie import visualize_gradients_flow_pie
from rnn.visualize.visualize_generate import visualize_generate


def run_visualizations(cost, updates,
                       train_stream, valid_stream,
                       args,
                       hidden_states=None, gate_values=None):

    # Load the parameters from a dumped model
    assert args.load_path is not None

    param_values = load_parameter_values(args.load_path)
    if args.hide_all_except is not None:
        i = args.hide_all_except
        sdim = args.state_dim

        output_size = get_output_size(args.dataset)

        hidden = np.zeros((args.layers * sdim, output_size), dtype=np.float32)

        output_w = param_values["/output_layer.W"]

        hidden[i * sdim: (i + 1) * sdim, :] = output_w[i * sdim:
                                                       (i + 1) * sdim, :]

        param_values["/output_layer.W"] = hidden

    model = Model(cost)
    model.set_parameter_values(param_values)

    # Run a visualization
    if args.visualize == "generate":
        visualize_generate(cost,
                           hidden_states, updates,
                           train_stream, valid_stream,
                           args)

    elif args.visualize == "gates" and (gate_values is not None):
        if args.rnn_type == "lstm":
            visualize_gates_lstm(gate_values, hidden_states, updates,
                                 train_stream, valid_stream,
                                 args)
        elif args.rnn_type == "soft":
            visualize_gates_soft(gate_values, hidden_states, updates,
                                 train_stream, valid_stream,
                                 args)
        else:
            assert False

    elif args.visualize == "states":
        visualize_states(hidden_states, updates,
                         train_stream, valid_stream,
                         args)

    elif args.visualize == "gradients":
        visualize_gradients(hidden_states, updates,
                            train_stream, valid_stream,
                            args)

    elif args.visualize == "jacobian":
        visualize_jacobian(hidden_states, updates,
                           train_stream, valid_stream,
                           args)

    elif args.visualize == "presoft":
        visualize_presoft(cost,
                          hidden_states, updates,
                          train_stream, valid_stream,
                          args)

    elif args.visualize == "matrices":
        visualize_matrices(args)

    elif args.visualize == "trained_singular_values":
        visualize_singular_values(args)

    elif args.visualize == "gradients_flow_pie":
        visualize_gradients_flow_pie(hidden_states, updates,
                                     args)

    else:
        assert False
