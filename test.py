from theano import tensor

from blocks.bricks import Initializable, Tanh
from blocks.bricks.base import application, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.roles import add_role, WEIGHT, INITIAL_STATE
from blocks.utils import shared_floatx_nans, shared_floatx_zeros

from collections import OrderedDict
import logging
import numpy
import theano

from blocks import initialization
from blocks.bricks import (Linear, Softmax,
                           FeedforwardSequence, MLP, Logistic,
                           Rectifier)
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import SimpleRecurrent, RecurrentStack

from bricks import LookupTable, HardLogistic

floatX = theano.config.floatX
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


class SoftGatedRecurrent(BaseRecurrent, Initializable):

    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, mlp=None,
                 **kwargs):
        super(SoftGatedRecurrent, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        self.activation = activation

        # The activation of the mlp should be a Logistic function
        self.mlp = mlp

        self.children = [activation, mlp]

    @property
    def state_to_state(self):
        return self.params[0]

    @property
    def matrix_gate(self):
        return self.params[1]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in ['inputs', 'states']:
            return self.dim
        return super(SoftGatedRecurrent, self).get_dim(name)

    def _allocate(self):
        self.params.append(shared_floatx_nans((self.dim, self.dim),
                                              name='state_to_state'))
        self.params.append(shared_floatx_zeros((self.dim,),
                                               name="initial_state"))
        add_role(self.params[0], WEIGHT)
        add_role(self.params[1], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)

    @recurrent(sequences=['mask', 'inputs'], states=['states'],
               outputs=['states', "gate_value"], contexts=[])
    def apply(self, inputs, states, mask=None):
        """Apply the gated recurrent transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim)
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.
        """
        # Concatenate the inputs of the MLP
        mlp_input = tensor.concatenate((inputs, states), axis=1)

        # Compute the output of the MLP
        gate_value = self.mlp.apply(mlp_input)

        # TODO: Find a way to remove the following "hack".
        # Simply removing the two next lines won't work
        gate_value = gate_value[:, 0]
        gate_value = gate_value[:, None]

        # Compute the next_states value, before gating
        next_states = self.activation.apply(
            states.dot(self.state_to_state) + inputs)

        # Apply the gating
        next_states = (next_states * gate_value +
                       states * (1 - gate_value))

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states, gate_value

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.params[2][None, :], batch_size, 0)]


def build_model_soft_test(vocab_size, args, dtype=floatX):
    logger.info('Building model ...')

    # Parameters for the model
    context = args.context
    state_dim = args.state_dim
    layers = args.layers
    skip_connections = args.skip_connections

    # Symbolic variables
    # In both cases: Time X Batch
    x = tensor.lmatrix('features')
    y = tensor.lmatrix('targets')

    # Build the model
    output_names = []
    output_dims = []
    for d in range(layers):
        if d > 0:
            suffix = '_' + str(d)
        else:
            suffix = ''
        if d == 0 or skip_connections:
            output_names.append("inputs" + suffix)
            output_dims.append(state_dim)

    lookup = LookupTable(length=vocab_size, dim=state_dim)
    lookup.weights_init = initialization.IsotropicGaussian(0.1)
    lookup.biases_init = initialization.Constant(0)

    fork = Fork(output_names=output_names, input_dim=args.mini_batch_size,
                output_dims=output_dims,
                prototype=FeedforwardSequence(
                    [lookup.apply]))

    transitions = [SimpleRecurrent(dim=state_dim, activation=Tanh())]

    # Build the MLP
    dims = [2 * state_dim]
    activations = []
    for i in range(args.mlp_layers):
        activations.append(Rectifier())
        dims.append(state_dim)

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

    for i in range(layers - 1):
        mlp = MLP(activations=activations, dims=dims,
                  weights_init=initialization.IsotropicGaussian(0.1),
                  biases_init=initialization.Constant(0),
                  name="mlp_" + str(i))
        transitions.append(
            SoftGatedRecurrent(dim=state_dim,
                               mlp=mlp,
                               activation=Tanh()))

    rnn = RecurrentStack(transitions, skip_connections=skip_connections)

    # dim = layers * state_dim
    output_layer = Linear(
        input_dim=layers * state_dim,
        output_dim=vocab_size, name="output_layer")

    # Return list of 3D Tensor, one for each layer
    # (Time X Batch X embedding_dim)
    pre_rnn = fork.apply(x)

    # Give a name to the input of each layer
    if skip_connections:
        for t in range(len(pre_rnn)):
            pre_rnn[t].name = "pre_rnn_" + str(t)
    else:
        pre_rnn.name = "pre_rnn"

    # Prepare inputs for the RNN
    kwargs = OrderedDict()
    init_states = {}
    for d in range(layers):
        if d > 0:
            suffix = '_' + str(d)
        else:
            suffix = ''
        if skip_connections:
            kwargs['inputs' + suffix] = pre_rnn[d]
        elif d == 0:
            kwargs['inputs' + suffix] = pre_rnn
        init_states[d] = theano.shared(
            numpy.zeros((args.mini_batch_size, state_dim)).astype(floatX),
            name='state0_%d' % d)
        kwargs['states' + suffix] = init_states[d]

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
    for d in range(layers):
        last_states[d] = h[d][-1, :, :]

    # Concatenate all the states
    if layers > 1:
        h = tensor.concatenate(h, axis=2)
    h.name = "hidden_state"

    # The updates of the hidden states
    updates = []
    for d in range(layers):
        updates.append((init_states[d], last_states[d]))

    presoft = output_layer.apply(h[context:, :, :])
    # Define the cost
    # Compute the probability distribution
    time, batch, feat = presoft.shape
    presoft.name = 'presoft'

    cross_entropy = Softmax().categorical_cross_entropy(
        y[context:, :].flatten(),
        presoft.reshape((batch * time, feat)))
    cross_entropy = cross_entropy / tensor.log(2)
    cross_entropy.name = "cross_entropy"

    # TODO: add regularisation for the cost
    # the log(1) is here in order to differentiate the two variables
    # for monitoring
    cost = cross_entropy + tensor.log(1)
    cost.name = "regularized_cost"

    # Initialize the model
    logger.info('Initializing...')

    fork.initialize()

    rnn.weights_init = initialization.Orthogonal()
    rnn.biases_init = initialization.Constant(0)
    rnn.initialize()

    output_layer.weights_init = initialization.IsotropicGaussian(0.1)
    output_layer.biases_init = initialization.Constant(0)
    output_layer.initialize()

    return cost, cross_entropy, updates, gate_values
