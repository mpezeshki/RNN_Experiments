import argparse
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='RNN_experiment')

    # Model options
    parser.add_argument('--rnn_type', choices=['lstm', 'simple', 'clockwork',
                                               'soft', 'hard'],
                        default='lstm')

    parser.add_argument('--layers', type=int,
                        default=4)
    parser.add_argument('--state_dim', type=int,
                        default=500)
    parser.add_argument('--skip_connections', action='store_true',
                        default=False)
    parser.add_argument('--skip_output', action="store_true",
                        default=False)
    parser.add_argument('--algorithm', choices=['rms_prop', 'adam', 'sgd'],
                        default='adam')

    # Options for the soft model
    parser.add_argument('--mlp_layers', type=int,
                        default=1)
    parser.add_argument('--mlp_activation', choices=['logistic',
                                                     'rectifier',
                                                     'hard_logistic'],
                        default="logistic")

    # Options for the clockwork
    parser.add_argument('--module_order', choices=["slow_in_fast",
                                                   "fast_in_slow"],
                        default="fast_in_slow")

    # Experiment options
    parser.add_argument('--dataset',
                        choices=['wikipedia', 'penntree',
                                 'mytext', 'wikipedia_junyoung', 'toy',
                                 'new_toy'],
                        default='wikipedia')
    parser.add_argument('--time_length', type=int,
                        default=150)
    parser.add_argument('--mini_batch_size', type=int,
                        default=5)
    parser.add_argument('--mini_batch_size_valid', type=int,
                        default=512)

    parser.add_argument('--context', type=int,
                        default=1)
    parser.add_argument('--tot_num_char', type=int,
                        default=None)
    parser.add_argument('--clipping', type=float,
                        default=5)
    parser.add_argument('--load_path', type=str,
                        default=None)
    parser.add_argument('--save_path', type=str,
                        default="/data/lisatmp3/zablocki/" +
                        "new_toy_4l_5units_simple_noskip_05_40")

    # Training options
    parser.add_argument('--learning_rate', type=float,
                        default=1e-3)
    parser.add_argument('--momentum', type=float,
                        default=0.9)

    # Regularization options
    parser.add_argument('--weight_noise', type=float,
                        default=0.0)

    # Monitoring options
    parser.add_argument('--generate', action="store_true", default=False)
    parser.add_argument('--initial_text_length', type=int,
                        default=10)
    parser.add_argument('--generated_text_lenght', type=int,
                        default=50)
    parser.add_argument('--patience', type=int,
                        default=20)
    parser.add_argument('--monitoring_freq', type=int,
                        default=500)
    parser.add_argument('--train_path', type=str,
                        default="/data/lisatmp3/zablocki/train.txt")
    parser.add_argument('--valid_path', type=str,
                        default="/data/lisatmp3/zablocki/valid.txt")
    parser.add_argument('--softmax_sampling', type=str,
                        choices=['random_sample', 'argmax'],
                        default='random_sample')

    # Visualization options
    parser.add_argument('--interactive_mode', action='store_true',
                        default=False)
    parser.add_argument('--visualize', choices=["nothing", "gates",
                                                "states", "gradients",
                                                "presoft", "matrices",
                                                "gradients_flow_pie",
                                                "trained_singular_values"],
                        default="nothing")
    parser.add_argument('--visualize_length', type=int,
                        default=75)
    parser.add_argument('--visualize_cells', action="store_true",
                        default=False)

    args = parser.parse_args()

    # Print all the arguments
    logger.info("\n" + "#" * 40)
    for arg in vars(args):
        logger.info('\"' + str(arg) + '\" \t: ' + str(vars(args)[arg]))
    logger.info("#" * 40 + "\n")

    return args


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


# 1 layer of 1000 LSTM = 4,250,000 parameters
# 1 layer of 2013 SIMPLE = 4,250,000 parameters
# 3 layers of 406 LSTM = 4,250,000 parameters
# 3 layers of 817 SIMPLE = 4,250,000 parameters
if __name__ == "__main__":
    param = compute_params(1000, 1, False, 50, "simple")
    param2 = compute_params(400, 3, True, 50, "simple")
    unit = compute_units(4250000, 1, False, 50, "simple")
    unit2 = compute_units(4250000, 3, True, 50, "simple")
    print(param)
    print(param2)
    print(unit)
    print(unit2)
