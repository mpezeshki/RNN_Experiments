import random
import string
import collections

import numpy as np


class GenerateXML(object):

    def __init__(self, depth, low_number, max_number):
        self.depth = depth
        self.low_number = low_number
        self.max_number = max_number

    def generate(self, length):
        # Initialize
        stack = []
        generated = []
        current_depth = 0.
        probability = np.array([1, 0.])
        string_length = 0
        best_score = 0

        for i in range(length):

            probability[0] = (self.depth - current_depth) / self.depth
            probability[1] = 1 - (self.depth - current_depth) / self.depth

            choice = np.random.choice(2, 1, p=probability)[0]

            # Open
            if choice == 0:
                current_depth += 1

                # Create tag
                tag_length = np.random.randint(
                    low_number, max_number, size=1)[0]

                tag = ''.join(
                    random.SystemRandom().choice(string.ascii_lowercase)
                    for _ in range(tag_length))

                # Add the tag to the stack
                stack.append(tag)

                # Add the tag to the generated
                generated.append("<" + tag + "> ")

                string_length += 3 + tag_length

            # Close recursion
            if choice == 1:
                current_depth -= 1
                tag = stack.pop()
                generated.append("</" + tag + "> ")

                string_length += 4 + len(tag)

        generated = "".join(generated)
        return generated, best_score, string_length


def get_vocab(text):
    counter = collections.Counter(text)
    freqs = collections.OrderedDict(dict(counter.most_common()))
    vocab = freqs.keys()
    return vocab


def string_parser(text, vocab):
    transormed_text = map(lambda char: vocab.index(char), text)
    return np.array(transormed_text).astype(np.int16)


def save(destination, train, valid, test, vocab):
    np.savez(destination,
             vocab=np.array(vocab),
             train=train,
             valid=valid,
             test=test,
             vocab_size=len(vocab))

if __name__ == "__main__":

    depth = 40.
    low_number = 2
    max_number = 10

    gen = GenerateXML(depth, low_number, max_number)

    # Train
    max_length = 1000000
    text, best_score, string_length = gen.generate(max_length)
    vocab = get_vocab(text)
    train = string_parser(text, vocab)

    # Valid
    max_length = 50000
    text, best_score, string_length = gen.generate(max_length)
    valid = string_parser(text, vocab)

    # Test
    max_length = 50000
    text, best_score, string_length = gen.generate(max_length)
    test = string_parser(text, vocab)

    save("/media/win/Users/Eloi/dataset/xml_tags/data",
         train,
         valid,
         test,
         vocab)
