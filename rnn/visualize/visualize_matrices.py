import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from blocks.serialization import load_parameter_values

path = "/media/win/Users/Eloi/new_toy_05_40_simple_4l_5units/best"

param_values = load_parameter_values(path)
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

pylab.show()