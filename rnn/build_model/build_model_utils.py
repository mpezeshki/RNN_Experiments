from collections import OrderedDict

import numpy

import theano
from theano import tensor

from blocks import initialization
from blocks.bricks import Linear, Softmax, FeedforwardSequence
from blocks.bricks.cost import SquaredError
from blocks.bricks.parallel import Fork
from rnn.datasets.dataset import has_indices, get_vocab_size, get_feature_size

from rnn.bricks import LookupTable

floatX = theano.config.floatX
RECURRENTSTACK_SEPARATOR = '#'


def get_prernn(args):

    # Compute the state dim
    if args.rnn_type == 'lstm':
        state_dim = 4 * args.state_dim
    else:
        state_dim = args.state_dim

    # Prepare the arguments for the fork
    output_names = []
    output_dims = []
    for d in range(args.layers):
        if d > 0:
            suffix = RECURRENTSTACK_SEPARATOR + str(d)
        else:
            suffix = ''
        if d == 0 or args.skip_connections:
            output_names.append("inputs" + suffix)
            output_dims.append(state_dim)

    # Prepare the brick to be forked (LookupTable or Linear)
    # Check if the dataset provides indices (in the case of a
    # fixed vocabulary, x is 2D tensor) or if it gives raw values
    # (x is 3D tensor)
    if has_indices(args.dataset):
        features = args.mini_batch_size
        x = tensor.lmatrix('features')
        vocab_size = get_vocab_size(args.dataset)
        lookup = LookupTable(length=vocab_size, dim=state_dim)
        lookup.weights_init = initialization.IsotropicGaussian(0.1)
        lookup.biases_init = initialization.Constant(0)
        forked = FeedforwardSequence([lookup.apply])

    else:
        x = tensor.tensor3('features', dtype=floatX)
        features = get_feature_size(args.dataset)
        forked = Linear(input_dim=features, output_dim=state_dim)
        forked.weights_init = initialization.IsotropicGaussian(0.1)
        forked.biases_init = initialization.Constant(0)

    # Define the fork
    fork = Fork(output_names=output_names, input_dim=features,
                output_dims=output_dims,
                prototype=forked)
    fork.initialize()

    # Apply the fork
    prernn = fork.apply(x)

    # Give a name to the input of each layer
    if args.skip_connections:
        for t in range(len(prernn)):
            prernn[t].name = "pre_rnn_" + str(t)
    else:
        prernn.name = "pre_rnn"

    return prernn


def get_presoft(h, args):
    if has_indices(args.dataset):
        vocab_size = get_vocab_size(args.dataset)
    else:
        vocab_size = get_feature_size(args.dataset)
    # If args.skip_connections: dim = args.layers * args.state_dim
    # else: dim = args.state_dim
    use_all_states = args.skip_connections or args.skip_output
    output_layer = Linear(
        input_dim=use_all_states * args.layers *
        args.state_dim + (1 - use_all_states) * args.state_dim,
        output_dim=vocab_size, name="output_layer")

    output_layer.weights_init = initialization.IsotropicGaussian(0.1)
    output_layer.biases_init = initialization.Constant(0)
    output_layer.initialize()
    presoft = output_layer.apply(h[args.context:, :, :])
    presoft.name = 'presoft'
    return presoft


def get_rnn_kwargs(pre_rnn, args):
    kwargs = OrderedDict()
    init_states = {}
    if args.rnn_type == 'lstm':
        init_cells = {}
    for d in range(args.layers):
        if d > 0:
            suffix = RECURRENTSTACK_SEPARATOR + str(d)
        else:
            suffix = ''
        if args.skip_connections:
            kwargs['inputs' + suffix] = pre_rnn[d]
        elif d == 0:
            kwargs['inputs'] = pre_rnn
        init_states[d] = theano.shared(
            numpy.zeros((args.mini_batch_size, args.state_dim)).astype(floatX),
            name='state0_%d' % d)
        if args.rnn_type == 'lstm':
            init_cells[d] = theano.shared(
                numpy.zeros((args.mini_batch_size,
                             args.state_dim)).astype(floatX),
                name='cell0_%d' % d)
        kwargs['states' + suffix] = init_states[d]
        if args.rnn_type == 'lstm':
            kwargs['cells' + suffix] = init_cells[d]
    inits = [init_states]
    if args.rnn_type == 'lstm':
        inits.append(init_cells)
    return kwargs, inits


def get_costs(presoft, args):

    if has_indices(args.dataset):
        # Targets: (Time X Batch)
        y = tensor.lmatrix('targets')

        time, batch, feat = presoft.shape
        cross_entropy = Softmax().categorical_cross_entropy(
            y[args.context:, :].flatten(),
            presoft.reshape((batch * time, feat)))

        unregularized_cost = cross_entropy / tensor.log(2)
        unregularized_cost.name = "cross_entropy"

    else:
        # Targets: (Time X Batch X Features)
        y = tensor.tensor3('targets', dtype=floatX)
        # Note: The target are one time step smaller that the features
        unregularized_cost = SquaredError().apply(presoft[:-1, :, :],
                                                  y[args.context:, :, :])
        unregularized_cost.name = "mean_squared_error"

    # TODO: add regularisation for the cost
    # the log(1) is here in order to differentiate the two variables
    # for monitoring
    cost = unregularized_cost + tensor.log(1)
    cost.name = "regularized_cost"
    return cost, unregularized_cost


def initialize_rnn(rnn, args):
    # Dont initialize as Orthogonal if we are about to load new parameters
    if args.load_path is not None:
        rnn.weights_init = initialization.Constant(0)
    else:
        rnn.weights_init = initialization.Orthogonal()
    rnn.biases_init = initialization.Constant(0)
    rnn.initialize()
