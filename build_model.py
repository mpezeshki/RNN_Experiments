import logging

import theano
from theano import tensor

from blocks import initialization
from blocks.bricks import Linear, Tanh, Softmax, Bias
from blocks.bricks.recurrent import LSTM, SimpleRecurrent
from blocks.bricks.lookup import LookupTable

floatX = theano.config.floatX
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def build_model(vocab_size, args, dtype=floatX):
    logger.info('Building model ...')

    context = args.context
    state_dim = args.state_dim
    rnn_type = args.rnn_type

    # Symbolic variables
    x = tensor.lmatrix('features')
    y = tensor.lmatrix('targets')

    # Build the model
    if rnn_type == "lstm":
        lookup = LookupTable(length=vocab_size, dim=4 * state_dim,
                             name='lookup')
        bias = Bias(4 * state_dim)
        rnn = LSTM(dim=state_dim, activation=Tanh())

    elif rnn_type == "simple":
        lookup = LookupTable(length=vocab_size, dim=state_dim, name='lookup')
        bias = Bias(state_dim)
        rnn = SimpleRecurrent(dim=state_dim, activation=Tanh())

    output_layer = Linear(
        input_dim=state_dim, output_dim=vocab_size, name="output_layer")

    # Return 3D Tensor: Batch X Time X embedding_dim
    pre_rnn = bias.apply(lookup.apply(x))

    # Give time as the first index: Time X Batch X embedding_dim
    pre_rnn = pre_rnn.dimshuffle(1, 0, 2)

    if rnn_type == "lstm":
        h = rnn.apply(pre_rnn)[0]

    elif rnn_type == "simple":
        h = rnn.apply(pre_rnn)

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
    rnn.initialize()

    output_layer.weights_init = initialization.IsotropicGaussian(0.1)
    output_layer.biases_init = initialization.Constant(0)
    output_layer.initialize()

    return cost, cross_entropy
