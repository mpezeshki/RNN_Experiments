import os
from collections import OrderedDict

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


def get_character():
    path = os.path.join(config.data_path, 'wikipedia-text',
                        'char_level_enwik8.npz')
    data = numpy.load(path)
    return data["vocab"]


def get_stream_char(which_set, time_length):
    path = os.path.join(config.data_path, 'wikipedia-text',
                        'char_level_enwik8.npz')
    data = numpy.load(path)
    train = data[which_set]

    dataset = IndexableDataset({'features': [train]})
    stream = DataStream(dataset, iteration_scheme=SequentialExampleScheme(1))
    stream = MakeRecurrent(time_length, stream)
    return stream


def get_minibatch_char(mini_batch_size, time_length):
    train_stream = get_stream_char("train", time_length)
    train_stream = Batch(train_stream,
                         iteration_scheme=ConstantScheme(mini_batch_size))
    valid_stream = get_stream_char("valid", time_length)
    valid_stream = Batch(valid_stream,
                         iteration_scheme=ConstantScheme(mini_batch_size))
    return train_stream, valid_stream

if __name__ == "__main__":
    # Test
    train_stream, valid_stream = get_minibatch_char(10, 200)
    print(next(train_stream.get_epoch_iterator()))
    print(next(valid_stream.get_epoch_iterator()))
