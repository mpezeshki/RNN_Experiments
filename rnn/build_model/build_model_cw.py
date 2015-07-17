import logging

import theano
from theano import tensor

from blocks import initialization
from blocks.bricks import Tanh
from blocks.bricks.recurrent import RecurrentStack

from rnn.bricks import ClockworkBase
from rnn.build_model.build_model_utils import (get_prernn, get_presoft,
                                               get_rnn_kwargs, get_costs)


floatX = theano.config.floatX
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def build_model_cw(vocab_size, args, dtype=floatX):
    logger.info('Building model ...')

    # In both cases: Time X Batch
    x = tensor.lmatrix('features')
    y = tensor.lmatrix('targets')

    # Return list of 3D Tensor, one for each layer
    # (Time X Batch X embedding_dim)
    pre_rnn = get_prernn(x, args)

    # Note that this order of the periods makes faster modules flow in slower
    # ones with is the opposite of the original paper
    if args.module_order == "fast_in_slow":
        transitions = [ClockworkBase(
            dim=args.state_dim, activation=Tanh(),
            period=2 ** i) for i in range(args.layers)]
    elif args.module_order == "slow_in_fast":
        transitions = [ClockworkBase(
            dim=args.state_dim,
            activation=Tanh(),
            period=2 ** (args.layers - i - 1)) for i in range(args.layers)]
    else:
        assert False

    rnn = RecurrentStack(transitions, skip_connections=args.skip_connections)

    # Prepare inputs and initial states for the RNN
    kwargs, inits = get_rnn_kwargs(pre_rnn, args)

    # Apply the RNN to the inputs
    h = rnn.apply(low_memory=True, **kwargs)

    # In the Clockwork case:
    # h = [state, time, state_1, time_1 ...]
    h = h[::2]

    # Now we have correctly:
    # h = [state, state_1, state_2 ...] if args.layers > 1
    # h = [state] if args.layers == 1

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
        h = tensor.concatenate(h, axis=2)
    else:
        h = h[0]
        last_states[0] = h[-1, :, :]
    h.name = "hidden_state_all"

    # The updates of the hidden states
    updates = []
    for d in range(args.layers):
        updates.append((inits[0][d], last_states[d]))

    presoft = get_presoft(h, args)

    cost, cross_entropy = get_costs(presoft, y, args)

    # Initialize the model
    logger.info('Initializing...')

    rnn.weights_init = initialization.Orthogonal()
    rnn.biases_init = initialization.Constant(0)
    rnn.initialize()

    return cost, cross_entropy, updates, hidden_states
