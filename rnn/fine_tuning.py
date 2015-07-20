import numpy as np

from blocks.model import Model
from blocks.serialization import load_parameter_values


def fine_tuning(cost, args):
    param_values = load_parameter_values(args.fine_tuning)

    param_values[
        "/output_layer.W"] = np.concatenate((
            param_values["/output_layer.W"],
            0.1 * np.random.randn(args.state_dim, 40).astype(np.float32)))

    model = Model(cost)
    model.set_parameter_values(param_values)

    return cost
