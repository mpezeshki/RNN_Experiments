import numpy as np
from numpy.random import randn
import theano

floatX = theano.config.floatX


def random_signal_lag(num_batches, batch_size, num_time_steps,
                      dim=1, lags=[2], noise=0.01):
    assert dim == len(lags)
    np.random.seed(0)

    input_seqs = randn(num_batches, num_time_steps,
                       batch_size, dim).astype(dtype=floatX)
    target_seqs = np.zeros((num_batches, num_time_steps, batch_size, dim),
                           dtype=floatX)

    for i, lag in enumerate(lags):
        target_seqs[:, lag:, :, i] = input_seqs[:, :-lag, :, i]
    target_seqs += noise * np.random.standard_normal(target_seqs.shape)

    return input_seqs, target_seqs


def sum_of_sines(num_batches, batch_size, num_time_steps,
                 depth=1):
    # Energy decrease when frequence increase:
    energies = np.array([1. / (2 * d + 1) for d in range(depth)])
    data = np.zeros((num_batches * num_time_steps, batch_size),
                    dtype=floatX)
    for i in range(batch_size):
        for d in range(depth):
            phase = np.random.uniform(low=-np.pi / 2,
                                      high=np.pi / 2,
                                      size=1)[0]
            frequency = np.random.uniform(low=1.2 * d + 0.3,
                                          high=1.2 * d + 0.6,
                                          size=1)[0]
            x = np.linspace(0, 5 * 2 * np.pi, num_batches * num_time_steps)

            energy = energies[d]

            sin = energy * np.sin(phase + frequency * x)

            data[:, i] += sin

        # Normalize the data
        # data /= np.max(np.abs(data), axis=0)
        import ipdb; ipdb.set_trace()
        # num_batches x num_time_steps x batch_size
        data2 = data.reshape((num_batches, num_time_steps, batch_size))
        # num_batches x batch_size x num_time_steps
        data2 = np.swapaxes(data2, 1, 2)

        data = data[:, :, None]
        return data
