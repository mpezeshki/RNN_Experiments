import logging

import numpy as np

import matplotlib.pyplot as plt
from rnn.datasets.dataset import conv_into_char, has_indices

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def plot(what, train_stream, compiled, args):
    # states
    epoch_iterator = train_stream.get_epoch_iterator()
    for num in range(10):
        init_ = next(epoch_iterator)[0][0: args.visualize_length, 0:1]

        values = compiled(init_)

        layers = len(values)
        time = values[0].shape[0]
        if has_indices(args.dataset):
            ticks = tuple(conv_into_char(init_[:, 0], args.dataset))
        else:
            ticks = tuple(np.arange(time))

        for d in range(layers):
            # Change the subplot
            plt.subplot(layers, 1, d + 1)

            # print only 5 values of the hiddenstate
            for j in range(10):
                plt.plot(np.arange(time), values[d][:, 0, j])
            # plt.plot(
            #     np.arange(time), np.mean(np.abs(values[d][:, 0, :]), axis=1))

            # Add ticks for xaxis
            plt.xticks(range(args.visualize_length), ticks)

            # Fancy options
            plt.grid(True)
            plt.title(what + "_of_layer_" + str(d))
        plt.tight_layout()

        # Either plot on the current display or save the plot into a file
        if args.local:
            plt.show()
        else:
            plt.savefig(
                args.save_path + "/visualize_" + what + '_' + str(num) + ".png")
            logger.info("Figure \"visualize_" + what + '_' + str(num) +
                        ".png\" saved at directory: " + args.save_path)
