import logging

import theano
from theano import tensor

from blocks import initialization
from blocks.bricks import Tanh, MLP, Logistic, Rectifier
from blocks.bricks.recurrent import SimpleRecurrent, RecurrentStack

from rnn.bricks import SoftGatedRecurrent, HardLogistic
from rnn.build_model.build_model_utils import (get_prernn, get_presoft,
                                               get_rnn_kwargs, get_costs,
                                               initialize_rnn)

floatX = theano.config.floatX
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def build_model_soft(args, dtype=floatX):
    logger.info('Building model ...')

    # Return list of 3D Tensor, one for each layer
    # (Time X Batch X embedding_dim)
    pre_rnn = get_prernn(args)

    transitions = [SimpleRecurrent(dim=args.state_dim, activation=Tanh())]

    # Build the MLP
    dims = [2 * args.state_dim]
    activations = []
    for i in range(args.mlp_args.layers):
        activations.append(Rectifier())
        dims.append(args.state_dim)

    # Activation of the last layer of the MLP
    if args.mlp_activation == "logistic":
        activations.append(Logistic())
    elif args.mlp_activation == "rectifier":
        activations.append(Rectifier())
    elif args.mlp_activation == "hard_logistic":
        activations.append(HardLogistic())
    else:
        assert False

    # Output of MLP has dimension 1
    dims.append(1)

    for i in range(args.layers - 1):
        mlp = MLP(activations=activations, dims=dims,
                  weights_init=initialization.IsotropicGaussian(0.1),
                  biases_init=initialization.Constant(0),
                  name="mlp_" + str(i))
        transitions.append(
            SoftGatedRecurrent(dim=args.state_dim,
                               mlp=mlp,
                               activation=Tanh()))

    rnn = RecurrentStack(transitions, skip_connections=args.skip_connections)
    initialize_rnn(rnn, args)

    # Prepare inputs and initial states for the RNN
    kwargs, inits = get_rnn_kwargs(pre_rnn, args)

    # Apply the RNN to the inputs
    h = rnn.apply(low_memory=True, **kwargs)

    # Now we have:
    # h = [state, state_1, gate_value_1, state_2, gate_value_2, state_3, ...]

    # Extract gate_values
    gate_values = h[2::2]
    new_h = [h[0]]
    new_h.extend(h[1::2])
    h = new_h

    # Now we have:
    # h = [state, state_1, state_2, ...]
    # gate_values = [gate_value_1, gate_value_2, gate_value_3]

    for i, gate_value in enumerate(gate_values):
        gate_value.name = "gate_value_" + str(i)

    # Save all the last states
    last_states = {}
    hidden_states = []
    for d in range(args.layers):
        last_states[d] = h[d][-1, :, :]
        h[d].name = "hidden_state_" + str(d)
        hidden_states.append(h[d])

    # Concatenate all the states
    if args.layers > 1:
        h = tensor.concatenate(h, axis=2)
    h.name = "hidden_state_all"

    # The updates of the hidden states
    updates = []
    for d in range(args.layers):
        updates.append((inits[0][d], last_states[d]))

    presoft = get_presoft(h, args)

    cost, cross_entropy = get_costs(presoft, args)

    return cost, cross_entropy, updates, gate_values, hidden_states
