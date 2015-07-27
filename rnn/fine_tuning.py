import numpy as np

from blocks.model import Model
from blocks.serialization import load_parameter_values
from rnn.datasets.dataset import get_output_size


def fine_tuning(cost, args):
    param_values = load_parameter_values(args.load_path)
    output_size = get_output_size(args.dataset)

    param_values[
        "/output_layer.W"] = np.concatenate((
            param_values["/output_layer.W"],
            0.1 * np.random.randn(args.state_dim,
                                  output_size).astype(np.float32)))

    model = Model(cost)
    model.set_parameter_values(param_values)

    return cost
