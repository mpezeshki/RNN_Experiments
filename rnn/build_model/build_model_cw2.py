import logging
import numpy as np

import theano

from blocks.bricks import Tanh

from rnn.bricks import ClockworkRecurrent
from rnn.build_model.build_model_utils import (get_prernn, get_presoft,
                                               get_rnn_kwargs, get_costs,
                                               initialize_rnn)


floatX = theano.config.floatX
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def build_model_cw2(args, dtype=floatX):
    logger.info('Building model ...')

    # Return list of 3D Tensor, one for each layer
    # (Time X Batch X embedding_dim)
    pre_rnn, x_mask = get_prernn(args)

    myrnn = ClockworkRecurrent(
        dim=args.state_dim,
        periods=list(2 ** np.array(range(args.cw_n))),
        activation=Tanh())

    rnn = myrnn
    initialize_rnn(rnn, args)

    # Prepare inputs and initial states for the RNN
    kwargs, inits = get_rnn_kwargs(pre_rnn, args)

    # Apply the RNN to the inputs
    h = rnn.apply(mask=x_mask, **kwargs)

    # Carrying last hidden state
    last_states = {}
    hidden_states = []
    # TODO correct bug
    # hidden_states.append(h * x_mask)
    hidden_states.append(h)
    hidden_states[0][0].name = "hidden_state_0"
    last_states[0] = h[0][-1, :, :]

    # The updates of the hidden states
    updates = []
    for d in range(args.layers):
        updates.append((inits[0][d], last_states[d]))

    presoft = get_presoft(h[0], args)

    cost, unregularized_cost = get_costs(presoft, args)

    return cost, unregularized_cost, updates, hidden_states
