import logging

import theano
from theano import tensor

from blocks.bricks import Tanh
from blocks.bricks.recurrent import RecurrentStack

from rnn.build_model.build_model_utils import (get_prernn, get_presoft,
                                               get_rnn_kwargs, get_costs,
                                               initialize_rnn)

from rnn.bricks import LSTM


floatX = theano.config.floatX
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def build_model_lstm(args, dtype=floatX):
    logger.info('Building model ...')

    # Return list of 3D Tensor, one for each layer
    # (Time X Batch X embedding_dim)
    pre_rnn, x_mask = get_prernn(args)

    transitions = [LSTM(dim=args.state_dim, activation=Tanh())
                   for _ in range(args.layers)]

    rnn = RecurrentStack(transitions, skip_connections=args.skip_connections)
    initialize_rnn(rnn, args)

    # Prepare inputs and initial states for the RNN
    kwargs, inits = get_rnn_kwargs(pre_rnn, args)

    # Apply the RNN to the inputs
    h = rnn.apply(mask=x_mask, **kwargs)

    # h = [state, cell, in, forget, out, state_1,
    #        cell_1, in_1, forget_1, out_1 ...]

    last_states = {}
    last_cells = {}
    hidden_states = []
    for d in range(args.layers):
        # TODO correct bug
        # h[5 * d] = h[5 * d] * x_mask
        # h[5 * d + 1] = h[5 * d + 1] * x_mask

        last_states[d] = h[5 * d][-1, :, :]
        last_cells[d] = h[5 * d + 1][-1, :, :]

        h[5 * d].name = "hidden_state_" + str(d)
        h[5 * d + 1].name = "hidden_cell_" + str(d)
        hidden_states.extend([h[5 * d], h[5 * d + 1]])

    # The updates of the hidden states
    # Note: if we have mask, then updating initial state
    # with last state does not make sence anymore.
    updates = []
    for d in range(args.layers):
        updates.append((inits[0][d], last_states[d]))
        updates.append((inits[1][d], last_states[d]))

    # h = [state, cell, in, forget, out, state_1,
    #        cell_1, in_1, forget_1, out_1 ...]

    # Extract the values
    in_gates = h[2::5]
    forget_gates = h[3::5]
    out_gates = h[4::5]

    gate_values = {"in_gates": in_gates,
                   "forget_gates": forget_gates,
                   "out_gates": out_gates}

    h = h[::5]

    # Now we have correctly:
    # h = [state, state_1, state_2 ...] if args.layers > 1
    # h = [state] if args.layers == 1

    # If we have skip connections, concatenate all the states
    # Else only consider the state of the highest layer
    if args.layers > 1:
        if args.skip_connections or args.skip_output:
            h = tensor.concatenate(h, axis=2)
        else:
            h = h[-1]
    else:
        h = h[0]
    h.name = "hidden_state_all"

    presoft = get_presoft(h, args)

    cost, unregularized_cost = get_costs(presoft, args)

    return cost, unregularized_cost, updates, gate_values, hidden_states
