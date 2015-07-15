import logging

import numpy as np

from blocks.serialization import load_parameter_values

import matplotlib
import matplotlib.pyplot as plt

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_matrices(args):
    if not args.local:
        # Force matplotlib to not use any Xwindows backend.
        matplotlib.use('Agg')

    param_values = load_parameter_values(args.load_path)
    print(param_values.keys())

    input0 = param_values["/fork/fork_inputs/lookuptable.W_lookup"]
    input1 = param_values["/fork/fork_inputs_1/lookuptable.W_lookup"]
    input2 = param_values["/fork/fork_inputs_2/lookuptable.W_lookup"]
    input3 = param_values["/fork/fork_inputs_3/lookuptable.W_lookup"]

    inputall = np.abs(np.concatenate((input0, input1, input2, input3), axis=1))

    outputall = np.abs(param_values["/output_layer.W"])

    # plt.matshow(inputall / np.mean(inputall), cmap=plt.cm.gray)
    # plt.matshow(outputall / np.mean(outputall), cmap=plt.cm.gray)
    plt.plot(np.arange(outputall.shape[0]), np.sum(np.abs(outputall), axis=1))

    plt.savefig(args.save_path + "/visualize_matrices.png")
    logger.info("Figure \"visualize_matrices"
                ".png\" saved at directory: " + args.save_path)
