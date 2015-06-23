import os

import numpy
from fuel import config
from fuel.datasets import IndexableDataset, TextFile
from fuel.schemes import SequentialExampleScheme, ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer, Batch


# Dictionaries for external text files
all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [',', '.', '!', '?', '<UNK>'] +
             [' ', '<S>', '</S>'])
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}


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


class MakeRecurrent2(Transformer):
    # Warning: code is hacky!
    def __init__(self, data_stream, target_source='targets'):
        if len(data_stream.sources) > 1:
            raise ValueError
        super(MakeRecurrent2, self).__init__(data_stream)
        self.sources = self.sources + (target_source,)
        self.sentence = []
        self.index = 0

    def get_data(self, request=None):
        self.sentence, = next(self.child_epoch_iterator)
        target = self.sentence[:, 1:self.sentence.shape[1] + 1]
        # to make the length of target equal to the length of input
        target = numpy.hstack([target, self.sentence[:, 0:1]])
        return (self.sentence, target)


def get_data(dataset):
    if dataset == "wikipedia":
        path = os.path.join(config.data_path, 'wikipedia-text',
                            'char_level_enwik8.npz')
    elif dataset == "penntree":
        path = os.path.join(config.data_path, 'PennTreebankCorpus',
                            'char_level_penntree.npz')
    return numpy.load(path, 'rb')


def get_character(dataset):
    if dataset not in ['wikipedia', 'penntree']:
        return numpy.array(all_chars)
    data = get_data(dataset)
    return data["vocab"]


def get_stream_char(dataset, which_set, time_length, total_num_chars=None):
    data = get_data(dataset)
    dataset = data[which_set]
    if total_num_chars is not None and which_set == 'train':
        dataset = dataset[:total_num_chars]
    else:
        total_num_chars = dataset.shape[0]

    dataset = IndexableDataset({'features': [dataset]})
    stream = DataStream(dataset, iteration_scheme=SequentialExampleScheme(1))
    stream = MakeRecurrent(time_length, stream)
    return stream, total_num_chars


def _lower(s):
    return s.lower()


def get_minibatch_char_for_text_file(mini_batch_size, time_length, args):
    dataset_options = dict(dictionary=char2code, level="character",
                           preprocess=_lower)
    train_dataset = TextFile([args.train_path], **dataset_options)
    train_stream = train_dataset.get_example_stream()
    train_stream = Batch(
        train_stream,
        iteration_scheme=ConstantScheme(mini_batch_size,
                                        args.tot_num_char))
    valid_dataset = TextFile([args.valid_path], **dataset_options)
    valid_stream = valid_dataset.get_example_stream()
    valid_stream = Batch(
        valid_stream,
        iteration_scheme=ConstantScheme(mini_batch_size,
                                        args.tot_num_char))
    vocab_size = len(all_chars)
    train_stream = MakeRecurrent2(train_stream)
    valid_stream = MakeRecurrent2(valid_stream)
    return train_stream, valid_stream, vocab_size


def get_minibatch_char(dataset, mini_batch_size,
                       time_length, args):
    if dataset not in ['wikipedia', 'penntree']:
        train_stream, valid_stream, vocab_size =\
            get_minibatch_char_for_text_file(mini_batch_size,
                                             time_length, args)
        return train_stream, valid_stream, vocab_size

    data = get_data(dataset)
    vocab_size = data['vocab_size']

    train_stream, train_num_examples = get_stream_char(
        dataset, "train", time_length, args.tot_num_char)
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
