import os
import re

import numpy
from fuel import config
from fuel.datasets import IndexableDataset
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer


def get_data(dataset):
    m = re.search("sine_(.+)", dataset)
    if dataset == "wikipedia":
        path = os.path.join(config.data_path, 'wikipedia-text',
                            'char_level_enwik8.npz')
    elif dataset == "wikipedia_junyoung":
        path = '/data/lisatmp3/zablocki/jun_data.npz'
    elif dataset == "penntree":
        path = os.path.join(config.data_path, 'PennTreebankCorpus',
                            'char_level_penntree.npz')
    elif dataset == "toy":
        path = os.path.join(config.data_path, 'toy_dependencies',
                            'new_05_40.npz')
    elif dataset == "xml":
        path = os.path.join(config.data_path, 'xml_tags',
                            'data.npz')
    elif dataset == "random":
        path = os.path.join(config.data_path, 'random_signal',
                            'data.npz')
    elif m:
        path = os.path.join(config.data_path, 'sine_waves',
                            'data_' + m.group(1) + '.npz')
    else:
        assert False
    return numpy.load(path, 'rb')


def has_indices(dataset):
    if dataset == "wikipedia":
        return True
    elif dataset == "wikipedia_junyoung":
        return True
    elif dataset == "penntree":
        return True
    elif dataset == "toy":
        return True
    elif dataset == "xml":
        return True
    elif dataset == "random":
        return False
    elif re.match("sine_", dataset):
        return False
    else:
        assert False


def get_output_size(dataset):
    data = get_data(dataset)
    if has_indices(dataset):
        return data["vocab_size"]
    else:
        return data["feature_size"]


def get_character(dataset):
    data = get_data(dataset)
    return data["vocab"]


def has_mask(dataset):
    data = get_data(dataset)
    return 'mask' in data.keys()


def conv_into_char(vector, dataset):
    correspondance = get_character(dataset)
    return correspondance[vector]


def get_stream_char(dataset, which_set, time_length, mini_batch_size,
                    total_train_chars=None):
    data = get_data(dataset)

    # dataset is one long string containing the whole sequence of indexes
    dataset = data[which_set]
    if total_train_chars is None:
        total_train_chars = dataset.shape[0]

    nb_mini_batches = total_train_chars / (mini_batch_size * time_length)
    total_train_chars = nb_mini_batches * mini_batch_size * time_length

    dataset = dataset[:total_train_chars]

    dataset = dataset.reshape(
        mini_batch_size, total_train_chars / mini_batch_size)
    dataset = dataset.T

    targets_dataset = dataset[1:, :]
    targets_dataset = numpy.concatenate(
        (targets_dataset,
         numpy.zeros((1, mini_batch_size)).astype(numpy.int64)), axis=0)

    dataset = dataset.reshape(
        nb_mini_batches,
        time_length, mini_batch_size)
    targets_dataset = targets_dataset.reshape(
        nb_mini_batches,
        time_length, mini_batch_size)

    dataset = IndexableDataset({'features': dataset,
                                'targets': targets_dataset})
    stream = DataStream(dataset,
                        iteration_scheme=SequentialExampleScheme(
                            nb_mini_batches))
    return stream


def get_stream_raw(dataset, which_set, mini_batch_size):
    data = get_data(dataset)

    # dataset is a 3D array of shape: Time X Batch X Features
    dataset = data[which_set]
    time, batch, features = dataset.shape
    nb_mini_batches = batch / mini_batch_size
    dataset = dataset[:, :nb_mini_batches * mini_batch_size, :]

    # Create the target_dataset
    targets_dataset = dataset[1:, :, :]

    # Cut the dataset into several minibatches
    # dataset is now 4D (nb_mini_batches X Time X mini_batch_size X Features)
    dataset = numpy.swapaxes(dataset, 0, 1)
    targets_dataset = numpy.swapaxes(targets_dataset, 0, 1)
    dataset = numpy.reshape(dataset, (nb_mini_batches, mini_batch_size,
                                      time, features))
    targets_dataset = numpy.reshape(targets_dataset, (nb_mini_batches,
                                                      mini_batch_size,
                                                      time - 1, features))
    dataset = numpy.swapaxes(dataset, 1, 2)
    targets_dataset = numpy.swapaxes(targets_dataset, 1, 2)

    # Create fuel dataset
    dataset = IndexableDataset({'features': dataset,
                                'targets': targets_dataset})
    stream = DataStream(dataset,
                        iteration_scheme=SequentialExampleScheme(
                            nb_mini_batches))
    return stream


def get_minibatch(args, total_train_chars=None):

    dataset = args.dataset
    mini_batch_size = args.mini_batch_size
    mini_batch_size_valid = args.mini_batch_size_valid
    time_length = args.time_length

    if has_indices(dataset):
        train_stream = get_stream_char(dataset, "train", time_length,
                                       mini_batch_size, total_train_chars)
        valid_stream = get_stream_char(dataset, "valid", time_length,
                                       mini_batch_size_valid,
                                       total_train_chars)
    else:
        train_stream = get_stream_raw(dataset, "train", mini_batch_size)
        valid_stream = get_stream_raw(dataset, "valid", mini_batch_size_valid)
    return train_stream, valid_stream


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial
    window_size = numpy.abs(numpy.int(window_size))
    order = numpy.abs(numpy.int(order))
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = numpy.mat([[k ** i for i in order_range]
                   for k in range(-half_window, half_window + 1)])
    m = numpy.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - numpy.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + numpy.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = numpy.concatenate((firstvals, y, lastvals))
    return numpy.convolve(m[::-1], y, mode='valid')


class BlurData(Transformer):

    def __init__(self, data_stream, target_source=0, **kwargs):

        super(BlurData, self).__init__(
            data_stream, **kwargs)
        self.sources = (self.sources[
            0] + "_" + str(target_source), "targets_" + str(target_source))

    def get_data(self, request=None):
        example = next(self.child_epoch_iterator)[0]

        blurred_example = numpy.zeros_like(example)
        for b in range(example.shape[1]):
            for f in range(example.shape[2]):
                blurred_example[:, b, f] = savitzky_golay(
                    example[:, b, f], 9, 3)
        target = example - blurred_example
        return (blurred_example, target)


if __name__ == "__main__":
    # Test

    class args(object):
        dataset = "sine_5"
        time_length = 300
        mini_batch_size = 5
        mini_batch_size_valid = 500

    train_stream, valid_stream = get_minibatch(args)

    blured = BlurData(train_stream)

    iterator = blured.get_epoch_iterator()
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))
