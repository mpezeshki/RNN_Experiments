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
                            'toy_dependencies.npz')
    elif dataset == "new_toy":
        path = os.path.join(config.data_path, 'toy_dependencies',
                            'new_05_40.npz')
    elif dataset == "xml":
        path = os.path.join(config.data_path, 'xml_tags',
                            'data.npz')
    else:
        assert False
    return numpy.load(path, 'rb')


def get_character(dataset):
    data = get_data(dataset)
    return data["vocab"]


def get_vocab_size(dataset):
    data = get_data(dataset)
    return data["vocab_size"]


def conv_into_char(vector, dataset):
    correspondance = get_character(dataset)
    return correspondance[vector]


def get_stream_char(dataset, which_set, time_length, mini_batch_size,
                    total_train_chars=None):
    data = get_data(dataset)
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
        total_train_chars / (mini_batch_size * time_length),
        time_length, mini_batch_size)
    targets_dataset = targets_dataset.reshape(
        total_train_chars / (mini_batch_size * time_length),
        time_length, mini_batch_size)
    # print dataset.shape
    # print targets_dataset.shape
    dataset = IndexableDataset({'features': dataset,
                                'targets': targets_dataset})
    stream = DataStream(dataset,
                        iteration_scheme=SequentialExampleScheme(
                            nb_mini_batches))
    # stream = MakeRecurrent(time_length, stream)
    return stream, total_train_chars


def get_minibatch_char(dataset, mini_batch_size, mini_batch_size_valid,
                       time_length, total_train_chars=None):
    data = get_data(dataset)
    vocab_size = data['vocab_size']

    train_stream, train_num_examples = get_stream_char(
        dataset, "train", time_length, mini_batch_size, total_train_chars)
    valid_stream, valid_num_examples = get_stream_char(
        dataset, "valid", time_length, mini_batch_size_valid)
    return train_stream, valid_stream, vocab_size

if __name__ == "__main__":
    # Test
    dataset = "penntree"
    time_length = 7
    mini_batch_size = 4

    voc = get_character(dataset)
    (train_stream,
        valid_stream,
        vocab_size) = get_minibatch_char(dataset,
                                         mini_batch_size,
                                         time_length)

    # print(voc)
    iterator = train_stream.get_epoch_iterator()
    print(next(iterator))
    print(next(iterator))
    print(next(iterator))
