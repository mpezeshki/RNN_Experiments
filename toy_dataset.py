import numpy as np
from numpy.random import choice

# open_mark = {1: "{", 2: "[", 3: "(", 4: '"', 5: "'"}
# close_mark = {1: "}", 2: "]", 3: ")", 4: 'a', 5: "b"}
# dictionary = {
#     "0": 0, "{": 1, "[": 2, "(": 3, '"': 4, "'": 5, "}": 6, "]": 7, ")": 8, 'a': 9, "b": 10}
probabilities = [0.4, 0.25, 0.15, 0.1, 0.05, 0.05]


class GenerateToy(object):

    def __init__(self, probabilities, length):
        self.probabilities = probabilities
        self.depth = len(probabilities)
        self.length = length

    def generate(self):
        ans = []
        length = 0
        while length < self.length:
            sample = choice(len(self.probabilities), 1, p=self.probabilities)
            sample = sample[0]
            # while (sample == 0):
            #     ans.append(0)
            #     length += 1
            #     sample = choice(
            #         len(self.probabilities), 1, p=self.probabilities)
            #     sample = sample[0]
            new_ans, new_length = self.recursive(sample)
            ans.extend(new_ans)
            length += new_length
        return ans, length

    # This is a recursive function
    def recursive(self, opened):
        sample = choice(len(self.probabilities), 1, p=self.probabilities)
        sample = sample[0]
        length = 2
        # ans = self.open_mark[opened]
        ans = [opened]
        while (sample != opened):
            if (sample < opened):
                # ans.append(0)
                # length += 1
                pass

            # Open a new depth of recursion
            else:
                (new_depth, new_length) = self.recursive(sample)
                ans.extend(new_depth)
                length += new_length

            sample = choice(len(self.probabilities), 1, p=self.probabilities)
            sample = sample[0]

        # Close the depth of recursion
        # return (ans + self.close_mark[sample], length)
        ans.append(opened + self.depth)
        return (ans, length)


def save(destination, train, valid, test):
    np.savez(destination,
             vocab=np.asarray(
                 ["x", "{", "[", "(", '"', "'", "}", "]", ")", "a", "b"]),
             train=train,
             valid=valid,
             test=test,
             vocab_size=11)

if __name__ == "__main__":

    # Train
    max_length = 300
    example = GenerateToy(probabilities, max_length)
    text, length = example.generate()
    text = np.asarray(text).astype(np.int16)
    train = text[:max_length]
    print train

    # max_length = 500000
    # example = GenerateToy(probabilities, max_length)
    # text, length = example.generate()
    # text = np.asarray(text).astype(np.int16)
    # valid = text[:max_length]

    # max_length = 500000
    # example = GenerateToy(probabilities, max_length)
    # text, length = example.generate()
    # text = np.asarray(text).astype(np.int16)
    # test = text[:max_length]

    # # save("/data/lisa/data/toy_dependencies/toy_dependencies",
    # save("/media/win/Users/Eloi/dataset/toy_dependencies/toy_dependencies",
    #      train,
    #      valid,
    #      test)
