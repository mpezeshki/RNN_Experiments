import numpy as np


class GenerateToy(object):

    def __init__(self, continue_prob, depth):
        self.continue_prob = continue_prob
        self.depth = depth

    def generate(self, length):
        stack = [0]
        generated = [0]
        best_score = 0
        probability = np.array(
            [1 - self.continue_prob, self.continue_prob, 0.])

        for i in range(length):

            probability[0] = (1 - self.continue_prob) * \
                (1 - stack[-1] / (self.depth - 1.))
            probability[2] = (1 - self.continue_prob) * \
                (stack[-1] / (self.depth - 1.))

            choice = np.random.choice(3, 1, p=probability)[0]

            # Continue
            if choice == 1:
                generated.append(stack[-1])
                best_score += np.log2(self.continue_prob)

            # Close recursion
            if choice == 2:
                stack.pop()
                generated.append(stack[-1])
                best_score += np.log2(probability[2])

            # Open a new depth of recursion
            if choice == 0:
                best_score += np.log2(probability[1] /
                                      (self.depth - stack[-1]))
                new_char = np.random.randint(
                    low=stack[-1], high=self.depth, size=1)[0]
                stack.append(new_char)
                generated.append(new_char)

        best_score = - best_score / float(length)
        return np.array(generated).astype(np.int16), best_score


def save(destination, train, valid, test, depth):
    np.savez(destination,
             vocab=str(np.arange(depth)),
             train=train,
             valid=valid,
             test=test,
             vocab_size=depth)

if __name__ == "__main__":

    continue_prob = 0.5
    depth = 40

    gen = GenerateToy(continue_prob, depth)

    # Train
    max_length = 400
    train, best_score1 = gen.generate(max_length)

    print train
    # # Valid
    # max_length = 500000
    # valid, best_score = gen.generate(max_length)

    # # Train
    # max_length = 500000
    # test, best_score = gen.generate(max_length)

    # save("/media/win/Users/Eloi/dataset/toy_dependencies/new_05_40",
    #      train,
    #      valid,
    #      test,
    #      depth)

    # print best_score1
