import numpy
from numpy.random import randn
import theano

floatX = theano.config.floatX


def random_signal_lag(num_batches, batch_size, num_time_steps,
                      dim=1, lags=[2], noise=0.01):
    assert dim == len(lags)
    numpy.random.seed(0)

    input_seqs = randn(num_batches, num_time_steps,
                       batch_size, dim).astype(dtype=floatX)
    target_seqs = numpy.zeros((num_batches, num_time_steps, batch_size, dim),
                              dtype=floatX)

    for i, lag in enumerate(lags):
        target_seqs[:, lag:, :, i] = input_seqs[:, :-lag, :, i]
    target_seqs += noise * numpy.random.standard_normal(target_seqs.shape)

    return input_seqs, target_seqs
