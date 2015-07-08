import theano
from theano import tensor

from blocks import initialization
from blocks.bricks import FeedforwardSequence
from blocks.bricks.parallel import Fork

from rnn.bricks import LookupTable
from rnn.datasets.dataset import get_minibatch_char
from rnn.utils import parse_args


def build_fork_lookup(vocab_size, args):
    x = tensor.lmatrix('features')
    virtual_dim = 6
    time_length = 5
    mini_batch_size = 2
    skip_connections = True
    layers = 3

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
            output_dims.append(virtual_dim)

    print output_names
    print output_dims
    lookup = LookupTable(length=vocab_size, dim=virtual_dim)
    lookup.weights_init = initialization.IsotropicGaussian(0.1)
    lookup.biases_init = initialization.Constant(0)

    fork = Fork(output_names=output_names, input_dim=time_length,
                output_dims=output_dims,
                prototype=FeedforwardSequence(
                    [lookup.apply]))

    # Return list of 3D Tensor, one for each layer
    # (Batch X Time X embedding_dim)
    pre_rnn = fork.apply(x)
    fork.initialize()

    f = theano.function([x], pre_rnn)
    return f

if __name__ == "__main__":
    args = parse_args()

    dataset = args.dataset

    mini_batch_size = 2
    time_length = 5

    # Prepare data
    train_stream, valid_stream, vocab_size = get_minibatch_char(
        dataset, mini_batch_size, time_length, args.tot_num_char)

    f = build_fork_lookup(vocab_size, args)
    data = next(train_stream.get_epoch_iterator())[1]
    print(data)
    print(f(data))
