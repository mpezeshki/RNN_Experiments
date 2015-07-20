import logging

import theano
from theano import tensor

from blocks.bricks import Tanh
from blocks.bricks.recurrent import SimpleRecurrent, RecurrentStack

from rnn.build_model.build_model_utils import (get_prernn, get_presoft,
                                               get_rnn_kwargs, get_costs,
                                               initialize_rnn)

floatX = theano.config.floatX
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def build_model_vanilla(vocab_size, args, dtype=floatX):
    logger.info('Building model ...')

    # Return list of 3D Tensor, one for each layer
    # (Time X Batch X embedding_dim)
    pre_rnn = get_prernn(args)

    transitions = [SimpleRecurrent(dim=args.state_dim, activation=Tanh())
                   for _ in range(args.layers)]

    rnn = RecurrentStack(transitions, skip_connections=args.skip_connections)
    initialize_rnn(rnn, args)

    # Prepare inputs and initial states for the RNN
    kwargs, inits = get_rnn_kwargs(pre_rnn, args)

    # Apply the RNN to the inputs
    h = rnn.apply(low_memory=True, **kwargs)

    # We have
    # h = [state, state_1, state_2 ...] if args.layers > 1
    # h = state if args.layers == 1

    # If we have skip connections, concatenate all the states
    # Else only consider the state of the highest layer
    last_states = {}
    hidden_states = []
    if args.layers > 1:
        # Save all the last states
        for d in range(args.layers):
            last_states[d] = h[d][-1, :, :]
            h[d].name = "hidden_state_" + str(d)
            hidden_states.append(h[d])
        if args.skip_connections or args.skip_output:
            h = tensor.concatenate(h, axis=2)
        else:
            h = h[-1]
    else:
        hidden_states.append(h)
        hidden_states[0].name = "hidden_state_0"
        last_states[0] = h[-1, :, :]

    # The updates of the hidden states
    updates = []
    for d in range(args.layers):
        updates.append((inits[0][d], last_states[d]))

    presoft = get_presoft(h, args)

    cost, cross_entropy = get_costs(presoft, args)

    return cost, cross_entropy, updates, hidden_states
