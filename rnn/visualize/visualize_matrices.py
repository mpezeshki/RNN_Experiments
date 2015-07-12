import logging

import numpy as np

from blocks.serialization import load_parameter_values

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pylab

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_matrices(args):

    param_values = load_parameter_values(args.save_path + "/best")
    print param_values.keys()

    input0 = param_values["/fork/fork_inputs/lookuptable.W_lookup"]
    input1 = param_values["/fork/fork_inputs_1/lookuptable.W_lookup"]
    input2 = param_values["/fork/fork_inputs_2/lookuptable.W_lookup"]
    input3 = param_values["/fork/fork_inputs_3/lookuptable.W_lookup"]

    inputall = np.concatenate((input0, input1, input2, input3), axis=1)

    outputall = param_values["/output_layer.W"]

    pylab.matshow(inputall, cmap=pylab.cm.gray)
    pylab.matshow(outputall, cmap=pylab.cm.gray)
    pylab.matshow(param_values["/output_layer.b"][None, :], cmap=pylab.cm.gray)

    pylab.tight_layout()
    pylab.savefig(args.save_path + "/visualize_matrices.png")
    logger.info("Figure \"visualize_matrices"
                ".png\" saved at directory: " + args.save_path)
