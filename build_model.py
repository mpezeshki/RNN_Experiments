from collections import OrderedDict
import logging

import theano
from theano import tensor

from blocks import initialization
from blocks.bricks import Linear, Tanh, Softmax, Bias
from blocks.bricks.recurrent import LSTM, SimpleRecurrent, RecurrentStack
from blocks.bricks.lookup import LookupTable

floatX = theano.config.floatX
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


# TODO: clean this function, split it in several pieces maybe
# TODO: handle skip_connections
def build_model(vocab_size, args, dtype=floatX):
    logger.info('Building model ...')

    # Parameters for the model
    context = args.context
    state_dim = args.state_dim
    rnn_type = args.rnn_type
    layers = args.layers
    skip_connections = args.skip_connections

    # Symbolic variables
    x = tensor.lmatrix('features')
    y = tensor.lmatrix('targets')

    # Build the model
    if rnn_type == "lstm":
        lookup = LookupTable(length=vocab_size, dim=4 * state_dim,
                             name='lookup')
        bias = Bias(4 * state_dim)
        transitions = [LSTM(dim=state_dim, activation=Tanh())
                       for _ in range(layers)]

    elif rnn_type == "simple":
        lookup = LookupTable(length=vocab_size, dim=state_dim, name='lookup')
        bias = Bias(state_dim)
        transitions = [SimpleRecurrent(dim=state_dim, activation=Tanh())
                       for _ in range(layers)]

    rnn = RecurrentStack(transitions, skip_connections=skip_connections)

    output_layer = Linear(
        input_dim=layers * state_dim,
        output_dim=vocab_size, name="output_layer")

    # Return 3D Tensor: Batch X Time X embedding_dim
    pre_rnn = bias.apply(lookup.apply(x))

    # Give time as the first index: Time X Batch X embedding_dim
    pre_rnn = pre_rnn.dimshuffle(1, 0, 2)

    # Prepare inputs for the RNN
    kwargs = OrderedDict()
    kwargs['inputs'] = pre_rnn

    # Apply the RNN to the inputs
    h = rnn.apply(low_memory=True, **kwargs)
    # h = [state_1, cell_1, state_2, cell_2 ...]

    if rnn_type == "lstm":
        h = h[::2]
    # h = [state_1, state_2, state_3 ...]

    if layers > 1:
        h = tensor.concatenate(h, axis=2)

    presoft = output_layer.apply(h[context:, :, :])
    # Define the cost
    # Compute the probability distribution
    time, batch, feat = presoft.shape
    presoft = presoft.dimshuffle(1, 0, 2)
    presoft = presoft.reshape((batch * time, feat))
    y = y[:, context:].flatten()

    cross_entropy = Softmax().categorical_cross_entropy(y, presoft)
    cross_entropy = cross_entropy / tensor.log(2)
    cross_entropy.name = "cross_entropy"

    # TODO: add regularisation for the cost
    # the log(1) is here in order to differentiate the two variables
    # for monitoring
    cost = cross_entropy + tensor.log(1)
    cost.name = "regularized_cost"

    # Initialize the model
    logger.info('Initializing...')

    lookup.weights_init = initialization.IsotropicGaussian(0.1)
    lookup.initialize()

    bias.biases_init = initialization.Constant(0)
    bias.initialize()

    rnn.weights_init = initialization.Orthogonal()
    rnn.biases_init = initialization.Constant(0)
    rnn.initialize()

    output_layer.weights_init = initialization.IsotropicGaussian(0.1)
    output_layer.biases_init = initialization.Constant(0)
    output_layer.initialize()

    return cost, cross_entropy
