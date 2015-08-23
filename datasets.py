import numpy as np
from numpy.random import randn
import theano
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from fuel import config
import os
from fuel.datasets import IndexableDataset
from fuel.schemes import SequentialExampleScheme

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

    train = IterableDataset({'x': input_seqs[:0.8 * num_batches],
                             'y': target_seqs[:0.8 * num_batches]})
    train_stream = DataStream(train)

    valid = IterableDataset({'x': input_seqs[0.8 * num_batches:],
                             'y': target_seqs[0.8 * num_batches:]})
    valid_stream = DataStream(valid)

    return train_stream, valid_stream


def sum_of_sines(num_batches, batch_size, num_time_steps,
                 depth=1):
    # Energy decrease when frequence increase:
    energies = np.array([1. / (2 * d + 1) for d in range(depth)])
    input_seqs = np.zeros((num_batches * num_time_steps, batch_size),
                          dtype=floatX)
    target_seqs = np.zeros((num_batches * num_time_steps, batch_size),
                           dtype=floatX)
    for i in range(batch_size):
        for d in range(depth):
            phase = np.random.uniform(low=-np.pi / 2,
                                      high=np.pi / 2,
                                      size=1)[0]
            frequency = np.random.uniform(low=1.2 * d + 0.3,
                                          high=1.2 * d + 0.6,
                                          size=1)[0]
            x = np.linspace(0, num_batches * 5 * np.pi,
                            num_batches * num_time_steps)

            energy = energies[d]

            sin = energy * np.sin(phase + frequency * x)

            input_seqs[:, i] += sin

    input_seqs /= np.max(np.abs(input_seqs), axis=0)

    # next step prediction!
    target_seqs[0:-1, :] = input_seqs[1:, :]
    # num_batches x num_time_steps x batch_size
    input_seqs = input_seqs.reshape((num_batches, num_time_steps, batch_size))
    target_seqs = target_seqs.reshape((num_batches, num_time_steps, batch_size))

    # S x T x B x F
    input_seqs = input_seqs[:, :, :, np.newaxis]
    target_seqs = target_seqs[:, :, :, np.newaxis]
    target_seqs += 0.0 * np.random.standard_normal(target_seqs.shape)

    train = IterableDataset({'x': input_seqs[:0.8 * num_batches],
                             'y': target_seqs[:0.8 * num_batches]})
    train_stream = DataStream(train)

    valid = IterableDataset({'x': input_seqs[0.8 * num_batches:],
                             'y': target_seqs[0.8 * num_batches:]})
    valid_stream = DataStream(valid)

    return train_stream, valid_stream


def get_stream_char(data, which_set, num_time_steps, batch_size):
    # dataset is one long string containing the whole sequence of indexes
    dataset = data[which_set]
    total_train_chars = dataset.shape[0]

    nb_mini_batches = total_train_chars / (batch_size * num_time_steps)
    total_train_chars = nb_mini_batches * batch_size * num_time_steps

    dataset = dataset[:total_train_chars]

    dataset = dataset.reshape(
        batch_size, total_train_chars / batch_size)
    dataset = dataset.T

    targets_dataset = dataset[1:, :]
    targets_dataset = np.concatenate(
        (targets_dataset,
         np.zeros((1, batch_size)).astype(np.int64)), axis=0)

    dataset = dataset.reshape(
        nb_mini_batches,
        num_time_steps, batch_size)
    targets_dataset = targets_dataset.reshape(
        nb_mini_batches,
        num_time_steps, batch_size)

    dataset = IndexableDataset({'features': dataset,
                                'targets': targets_dataset})
    stream = DataStream(dataset,
                        iteration_scheme=SequentialExampleScheme(
                            nb_mini_batches))
    return stream


def wikipedia(batch_size, num_time_steps):
    path = os.path.join(config.data_path, 'wikipedia-text',
                        'char_level_enwik8.npz')
    data = np.load(path, 'rb')

    train_stream = get_stream_char(data, "train", num_time_steps,
                                   batch_size)
    valid_stream = get_stream_char(data, "valid", num_time_steps,
                                   batch_size)
    return train_stream, valid_stream
