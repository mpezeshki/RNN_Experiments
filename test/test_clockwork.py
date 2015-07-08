from collections import OrderedDict

import theano
from theano import tensor
from bricks import LookupTable, ClockworkBase
from blocks.bricks import FeedforwardSequence, Tanh
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import RecurrentStack, SimpleRecurrent
from blocks import initialization

from dataset import get_minibatch_char
from utils import parse_args


def build_fork_lookup(vocab_size, time_length, args):
    x = tensor.lmatrix('features')
    virtual_dim = 6
    state_dim = 6
    skip_connections = False
    layers = 1

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

    lookup = LookupTable(length=vocab_size, dim=virtual_dim)
    lookup.weights_init = initialization.IsotropicGaussian(0.1)
    lookup.biases_init = initialization.Constant(0)

    fork = Fork(output_names=output_names, input_dim=time_length,
                output_dims=output_dims,
                prototype=FeedforwardSequence(
                    [lookup.apply]))

    # Note that this order of the periods makes faster modules flow in slower
    # ones with is the opposite of the original paper
    transitions = [ClockworkBase(dim=state_dim, activation=Tanh(),
                                 period=2 ** i) for i in range(layers)]

    rnn = RecurrentStack(transitions, skip_connections=skip_connections)

    # Return list of 3D Tensor, one for each layer
    # (Batch X Time X embedding_dim)
    pre_rnn = fork.apply(x)

    # Give time as the first index for each element in the list:
    # (Time X Batch X embedding_dim)
    if layers > 1 and skip_connections:
        for t in range(len(pre_rnn)):
            pre_rnn[t] = pre_rnn[t].dimshuffle(1, 0, 2)
    else:
        pre_rnn = pre_rnn.dimshuffle(1, 0, 2)

    f_pre_rnn = theano.function([x], pre_rnn)

    # Prepare inputs for the RNN
    kwargs = OrderedDict()
    for d in range(layers):
        if d > 0:
            suffix = '_' + str(d)
        else:
            suffix = ''
        if d == 0 or skip_connections:
            if skip_connections:
                kwargs['inputs' + suffix] = pre_rnn[d]
            else:
                kwargs['inputs' + suffix] = pre_rnn

    print kwargs
    # Apply the RNN to the inputs
    h = rnn.apply(low_memory=True, **kwargs)

    fork.initialize()

    rnn.weights_init = initialization.Orthogonal()
    rnn.biases_init = initialization.Constant(0)
    rnn.initialize()

    f_h = theano.function([x], h)
    return f_pre_rnn, f_h

if __name__ == "__main__":
    args = parse_args()

    dataset = args.dataset

    mini_batch_size = 2
    time_length = 10

    # Prepare data
    train_stream, valid_stream, vocab_size = get_minibatch_char(
        dataset, mini_batch_size, time_length, args.tot_num_char)

    f_pre_rnn, f_h = build_fork_lookup(vocab_size, time_length, args)
    data = next(train_stream.get_epoch_iterator())[1]
    print(data)

    pre_rnn = f_pre_rnn(data)
    h = f_h(data)

    print pre_rnn.shape
    # print pre_rnn[1].shape
    print h[0].shape
    print h[1].shape
    print h[0]
    print h[1]
    # print h[2].shape
    # print h[3].shape
    # print h[3]
