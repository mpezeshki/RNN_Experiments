import numpy as np
import matplotlib.pyplot as plt


class GenerateSineWave(object):

    def __init__(self, depth, time):
        self.depth = depth
        self.time = time

    def generate(self, batch):
        data = np.zeros((self.time, batch), dtype=np.float32)
        for i in range(batch):
            for d in range(depth):
                phase = np.random.randn(1)[0]
                sin = np.sin(phase +
                             (d + 1) * np.linspace(0, 4 * np.pi, self.time))
                data[:, i] += sin / (2 * d + 1)

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
    depth = 1
    time = 150
    generator = GenerateSineWave(depth, time)

    # Train
    batch = 100000
    train = generator.generate(batch)

    # Valid
    batch = 10000
    valid = generator.generate(batch)

    # Test
    batch = 10000
    test = generator.generate(batch)

    # Save the data
    save("/media/win/Users/Eloi/dataset/sine_waves/data_d1",
         train,
         valid,
         test)

    for i in range(batch):
        plt.plot(range(time), train[:, i, 0])
    plt.grid()
    plt.show()
