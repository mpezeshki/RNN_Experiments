# Compute the number of parameters given a particular architecture
def compute_params(dim, layers, skip_connections, vocab_size, unit_type):
    virtual_dim = dim
    if unit_type == 'lstm':
        virtual_dim = 4 * dim

    if skip_connections:
        input_to_layers = virtual_dim * vocab_size * layers
        between_layers = 0.5 * layers * (layers - 1) * virtual_dim * dim
    else:
        input_to_layers = virtual_dim * vocab_size
        between_layers = (layers - 1) * virtual_dim * dim

    in_layers = virtual_dim * dim * layers
    output = layers * dim * vocab_size

    param = input_to_layers + in_layers + between_layers + output

    return param


# Stupidely compute the number of units in each layer given the total number
# of parameters allowed
def compute_units(param, layers, skip_connections, vocab_size, unit_type):
    unit = 1
    while(compute_params(
            unit, layers, skip_connections, vocab_size, unit_type) < param):
        unit += 1
    return unit


if __name__ == "__main__":
    param = compute_params(1000, 1, False, 50, "lstm")
    unit = compute_units(4000000, 3, True, 50, "lstm")
    print(param)
    print(unit)
