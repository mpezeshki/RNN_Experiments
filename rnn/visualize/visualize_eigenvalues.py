from scipy.linalg import svd

from blocks.serialization import load_parameter_values


def visualize_eigenvalues(args):
    path = args.load_path
    layers = args.layers

    param_values = load_parameter_values(path)
    print(param_values.keys())

    matrices = {}
    for i in range(layers):
        matrices[i] = param_values[
            "/recurrentstack/simplerecurrent_" + str(i) + ".W"]
        u, diag, v = svd(matrices[i])
        print(diag)
