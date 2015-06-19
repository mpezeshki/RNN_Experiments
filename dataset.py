import os

import numpy
from fuel import config
from fuel.datasets import IndexableDataset
from fuel.schemes import SequentialExampleScheme, ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer, Batch


class MakeRecurrent(Transformer):

    def __init__(self, time_length, data_stream, target_source='targets'):
        if len(data_stream.sources) > 1:
            raise ValueError
        super(MakeRecurrent, self).__init__(data_stream)
        self.sources = self.sources + (target_source,)
        self.time_length = time_length
        self.sentence = []
        self.index = 0

    def get_data(self, request=None):
        while not self.index < len(self.sentence) - self.time_length - 1:
            self.sentence, = next(self.child_epoch_iterator)
            self.index = 0
        x = self.sentence[self.index:self.index + self.time_length]
        target = self.sentence[
            self.index + 1:self.index + self.time_length + 1]
        self.index += self.time_length
        return (x, target)


def get_data(dataset):
    if dataset == "wikipedia":
        path = os.path.join(config.data_path, 'wikipedia-text',
                            'char_level_enwik8.npz')
    elif dataset == "penntree":
        path = os.path.join(config.data_path, 'PennTreebankCorpus',
                            'char_level_penntree.npz')
    return numpy.load(path, 'rb')


def get_character(dataset):
    data = get_data(dataset)
    return data["vocab"]


def get_stream_char(dataset, which_set, time_length, total_train_chars=None):
    data = get_data(dataset)
    dataset = data[which_set]
    if total_train_chars is not None:
        dataset = dataset[:total_train_chars]
    else:
        total_train_chars = dataset.shape[0]

    dataset = IndexableDataset({'features': [dataset]})
    stream = DataStream(dataset, iteration_scheme=SequentialExampleScheme(1))
    stream = MakeRecurrent(time_length, stream)
    return stream, total_train_chars


def get_minibatch_char(dataset, mini_batch_size,
                       time_length, total_train_chars=None):
    data = get_data(dataset)
    vocab_size = data['vocab_size']

    train_stream, train_num_examples = get_stream_char(
        dataset, "train", time_length, total_train_chars)
    train_stream = Batch(
        train_stream,
        iteration_scheme=ConstantScheme(mini_batch_size, train_num_examples))
    valid_stream, valid_num_examples = get_stream_char(
        dataset, "valid", time_length)
    valid_stream = Batch(
        valid_stream,
        iteration_scheme=ConstantScheme(mini_batch_size, valid_num_examples))
    return train_stream, valid_stream, vocab_size

if __name__ == "__main__":
    # Test
    dataset = "penntree"
    voc = get_character(dataset)
    train_stream, valid_stream, vocab_size = get_minibatch_char(dataset,
                                                                10, 200)

    print(voc)
    print(next(train_stream.get_epoch_iterator()))
    print(next(valid_stream.get_epoch_iterator()))
