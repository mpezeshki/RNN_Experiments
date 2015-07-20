import logging

import numpy as np

from blocks.serialization import load_parameter_values

import matplotlib.pyplot as plt

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_singular_values(args):
    param_values = load_parameter_values(args.load_path)
    for d in range(args.layers):
        if args.rnn_type == 'lstm':
            ws = param_values["/recurrentstack/lstm_" + str(d) + ".W_state"]
            w_rec = ws[:, 3 * args.state_dim:]
        elif args.rnn_type == 'simple':
            w_rec = param_values["/recurrentstack/simplerecurrent_" + str(d) +
                                 ".W_state"]
        else:
            raise NotImplementedError
        U, s, V = np.linalg.svd(w_rec, full_matrices=True)
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(s.shape[0]), s, label='Layer_' + str(d))
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.title("Singular_values_of_recurrent_weights")
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(s.shape[0]), np.log(s + 1E-15),
                 label='Layer_' + str(d))
        plt.grid(True)
        plt.title("Log_singular_values_of_recurrent_weights")
    plt.tight_layout()

    plt.savefig(args.save_path + "/visualize_singular_values.png")
    logger.info("Figure \"visualize_singular_values"
                ".png\" saved at directory: " + args.save_path)
