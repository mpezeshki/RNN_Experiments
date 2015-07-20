import numpy as np


class GenerateSineWave(object):

    def __init__(self, depth, time):
        self.depth = depth
        self.time = time

    def generate(self, batch):
        data = np.zeros((self.time, batch), dtype=np.float32)
        for i in range(batch):
            for d in range(depth):
                phase = np.random.randn(1)[0]
                data[:, i] += np.sin(phase + (d + 1) *
                                     np.linspace(0, 3 * np.pi, self.time))

        # Center and normalize the data
        data -= np.mean(data, axis=0)
        data /= np.max(np.abs(data), axis=0)

        data = data[:, :, None]
        return data


def save(destination, train, valid, test):
    np.savez(destination,
             train=train,
             valid=valid,
             test=test,
             feature_size=1)


if __name__ == "__main__":
    depth = 5
    time = 150
    generator = GenerateSineWave(depth, time)

    # Train
    batch = 1000
    train = generator.generate(batch)

    # Valid
    batch = 1000
    valid = generator.generate(batch)

    # Test
    batch = 1000
    test = generator.generate(batch)

    # Save the data
    save("/media/win/Users/Eloi/dataset/sine_waves/datatmp", train, valid, test)
