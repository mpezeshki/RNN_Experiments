import numpy as np


class OptimalScore(object):

    def compute_prob(self, i):
        self.probability[0] = 0
        for j in range(i):
            self.probability[0] += self.prior[j]
            self.probability[j + 1] = 0

        for j in range(5):
            self.probability[j + 6] = 0
        self.probability[i + 5] = self.prior[i]

        for j in range(i + 1, 6):
            self.probability[j] = self.prior[j]
        return self.probability

    def compute(self, text, prior):
        self.prior = prior
        self.probability = np.copy(prior)
        stack_depth = []
        count = 0
        for i in text:
            count += np.log2(self.probability[i])
            if i == 0:
                continue
            elif (i <= 5):
                self.probability = self.compute_prob(i)
                stack_depth.append(i)

            elif (i > 5):
                assert i == stack_depth.pop() + 5
                if len(stack_depth) > 0:
                    last_oppened = stack_depth.pop()
                    stack_depth.append(last_oppened)
                    self.probability = self.compute_prob(last_oppened)
                else:
                    self.probability = np.copy(self.prior)
            else:
                assert False

            assert np.sum(self.probability) == 1.
        return - count / float(text.shape[0])


class BiasedScore(object):

    def compute(self, text):
        probability = np.zeros((11,))
        for i in text:
            probability[i] += 1
        probability /= valid.shape[0]

        count = 0
        for i in text:
            count += np.log2(probability[i])

        return - count / float(text.shape[0])

if __name__ == "__main__":
    path = "/media/win/Users/Eloi/dataset/toy_dependencies/toy_dependencies.npz"
    data = np.load(path)
    valid = data["train"]
    prior = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02, 0., 0., 0., 0., 0.])
    biased_prior = np.array(
        [0.9, 0.000000001, 0.005, 0.02, 0.035, 0.04 - 0.000000001, 0., 0., 0., 0., 0.])
    optimal = OptimalScore()
    print optimal.compute(valid, biased_prior)

    # biased = BiasedScore()
    # print biased.compute(valid)
