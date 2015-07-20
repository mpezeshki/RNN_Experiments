import os

import numpy
from fuel import config
from fuel.datasets import IndexableDataset
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream


def get_data(dataset):
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
    elif dataset == "sine":
        path = os.path.join(config.data_path, 'sine_waves',
                            'data.npz')
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
    elif dataset == "sine":
        return False
    else:
        assert False


def get_character(dataset):
    data = get_data(dataset)
    return data["vocab"]


def get_vocab_size(dataset):
    data = get_data(dataset)
    return data["vocab_size"]


def get_feature_size(dataset):
    data = get_data(dataset)
    return data["feature_size"]


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

    # Create the target_dataset
    targets_dataset = dataset[1:, :, :]

    # Cut the dataset into several minibatches
    # dataset is now 4D (nb_mini_batches X Time X mini_batch_size X Features)
    dataset = numpy.swapaxes(dataset, 0, 1)
    targets_dataset = numpy.swapaxes(targets_dataset, 0, 1)
    dataset = numpy.reshape(dataset, (nb_mini_batches, mini_batch_size,
                                      time, features))
    dataset = numpy.reshape(dataset, (nb_mini_batches, mini_batch_size,
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


def get_minibatch(dataset, mini_batch_size, mini_batch_size_valid,
                  time_length, total_train_chars=None):

    if has_indices(dataset):
        train_stream = get_stream_char(dataset, "train", time_length,
                                       mini_batch_size, total_train_chars)
        valid_stream = get_stream_char(dataset, "valid", time_length,
                                       mini_batch_size_valid,
                                       total_train_chars)
    else:
        train_stream = get_stream_raw(dataset, "train", mini_batch_size)
        valid_stream = get_stream_raw(dataset, "valid", mini_batch_size)
    return train_stream, valid_stream

if __name__ == "__main__":
    # Test
    dataset = "xml"
    time_length = 7
    mini_batch_size = 4
    mini_batch_size_valid = 20

    voc = get_character(dataset)
    train_stream, valid_stream = get_minibatch(dataset,
                                               mini_batch_size,
                                               mini_batch_size_valid,
                                               time_length)

    # print(voc)
    iterator = train_stream.get_epoch_iterator()
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))
