import matplotlib.pylab as plt
import numpy as np


# TODO normalize data
# TODO check the datastream
class GenerateSineWave(object):

    def __init__(self, depth, example_length):
        self.depth = depth
        self.example_length = example_length

    def generate(self, length):
        data = np.zeros((0), dtype=np.float32)
        for i in range(length):
            new_example = np.zeros((self.example_length,), dtype=np.float32)
            for d in range(depth):
                phase = np.random.randn(1)[0]
                new_example += np.sin(phase + (d + 1) *
                                      np.linspace(0, 2 * np.pi,
                                                  self.example_length))
            data = np.concatenate((data, new_example), axis=1)
        return data


if __name__ == "__main__":
    depth = 5
    example_length = 150
    length = 3

    generator = GenerateSineWave(depth, example_length)
    data = generator.generate(length)

    plt.plot(range(length * example_length), data)
    plt.show()
