from collections import OrderedDict
import logging

import theano
from theano import tensor

from blocks import initialization
from blocks.bricks import Linear, Tanh, Softmax, FeedforwardSequence
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import SimpleRecurrent, RecurrentStack

from bricks import LookupTable

floatX = theano.config.floatX
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


# TODO: clean this function, split it in several pieces maybe
def build_model_soft(vocab_size, args, dtype=floatX):
    logger.info('Building model ...')

    # Parameters for the model
    context = args.context
    state_dim = args.state_dim
    layers = args.layers
    time_length = args.time_length

    # Symbolic variables
    x = tensor.lmatrix('features')
    y = tensor.lmatrix('targets')

    # Build the model
    output_names = ["inputs"]
    output_dims = [state_dim]

    lookup = LookupTable(length=vocab_size, dim=state_dim)
    lookup.weights_init = initialization.IsotropicGaussian(0.1)
    lookup.biases_init = initialization.Constant(0)

    fork = Fork(output_names=output_names, input_dim=time_length,
                output_dims=output_dims,
                prototype=FeedforwardSequence(
                    [lookup.apply]))

    transitions = [SimpleRecurrent(dim=state_dim, activation=Tanh())
                   for _ in range(layers)]

    rnn = RecurrentStack(transitions, skip_connections=False)

    # dim = layers * state_dim
    output_layer = Linear(
        input_dim=layers * state_dim,
        output_dim=vocab_size, name="output_layer")

    # Return list of 3D Tensor, one for each layer
    # (Batch X Time X embedding_dim)
    pre_rnn = fork.apply(x)

    # Give time as the first index for each element in the list:
    # (Time X Batch X embedding_dim)

    for t in range(len(pre_rnn)):
        pre_rnn[t] = pre_rnn[t].dimshuffle(1, 0, 2)

    # Prepare inputs for the RNN
    kwargs = OrderedDict()
    for d in range(layers):
        if d > 0:
            suffix = '_' + str(d)
        else:
            suffix = ''
        if d == 0:
            kwargs['inputs' + suffix] = pre_rnn

    # Apply the RNN to the inputs
    h = rnn.apply(low_memory=True, **kwargs)

    # Now we have correctly:
    # h = [state_1, state_2, state_3 ...]

    # Concatenate all the states
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

    fork.initialize()

    rnn.weights_init = initialization.IsotropicGaussian(0.1)
    rnn.biases_init = initialization.Constant(0)
    rnn.initialize()

    output_layer.weights_init = initialization.IsotropicGaussian(0.1)
    output_layer.biases_init = initialization.Constant(0)
    output_layer.initialize()

    return cost, cross_entropy
