import numpy as np
import matplotlib.pyplot as plt


class GenerateSineWave(object):

    def __init__(self, depth, time, energies):
        self.depth = depth
        self.time = time
        self.energies = energies

    def generate(self, batch):
        data = np.zeros((self.time, batch), dtype=np.float32)
        for i in range(batch):
            for d in range(depth):
                phase = np.random.uniform(low=-np.pi / 2,
                                          high=np.pi / 2,
                                          size=1)[0]
                frequency = np.random.uniform(low=1.2 * d + 0.3,
                                              high=1.2 * d + 0.6,
                                              size=1)[0]
                x = np.linspace(0, 5 * 2 * np.pi, self.time)

                energy = self.energies[d]

                sin = energy * np.sin(phase + frequency * x)

                data[:, i] += sin

        # Normalize the data
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
    depth = 6
    time = 300

    # Energy decrease when frequence increase:
    energies1 = np.array([1. / (2 * d + 1) for d in range(depth)])

    # Energy increases when frequence increase:
    energies2 = np.array([1. / (2 * (depth - d)) for d in range(depth)])

    # Energy decrease when frequence increase:
    energies3 = np.array([1. / depth for d in range(depth)])

    generator = GenerateSineWave(depth, time, energies3)

    # Train
    batch = 1
    train = generator.generate(batch)

    # # Valid
    # batch = 10000
    # valid = generator.generate(batch)

    # # Test
    # batch = 10000
    # test = generator.generate(batch)

    # # Save the data
    # save("/media/win/Users/Eloi/dataset/sine_waves/data_4",
    #      train,
    #      valid,
    #      test)

    for i in range(batch):
        plt.plot(range(time), train[:, i, 0])
    axes = plt.gca()
    axes.set_ylim([-1, 1])
    plt.grid()
    plt.show()
