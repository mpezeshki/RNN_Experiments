import argparse
import logging
import numpy
import theano

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='RNN_experiment')

    # Model options
    parser.add_argument('--rnn_type', choices=['lstm', 'simple', 'clockwork',
                                               'clockwork2', 'soft', 'hard'],
                        default='lstm')

    parser.add_argument('--layers', type=int,
                        default=1)
    parser.add_argument('--state_dim', type=int,
                        default=20)
    parser.add_argument('--skip_connections', action='store_true',
                        default=False)
    parser.add_argument('--skip_output', action="store_true",
                        default=False)
    parser.add_argument('--algorithm', choices=['rms_prop', 'adam', 'sgd'],
                        default='adam')
    parser.add_argument('--non_linearity', choices=["tanh", "relu"],
                        default="tanh")

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
    # Number of components in clockwork2
    parser.add_argument('--cw_n', type=int, default=1)

    # Experiment options

    # Choices = ['wikipedia', 'penntree', 'mytext', wikipedia_junyoung', 'toy',
    # 'xml', 'sine_1', 'sine_2...]
    parser.add_argument('--dataset', type=str,
                        default='sine_1')
    parser.add_argument('--time_length', type=int,
                        default=300)
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
    parser.add_argument('--used_inputs', type=int,
                        default=None)
    parser.add_argument('--orthogonal_init', action="store_true",
                        default=False)
    parser.add_argument('--fine_tuning', action="store_true",
                        default=False)

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
                        default=50)
    parser.add_argument('--generated_text_lenght', type=int,
                        default=100)
    parser.add_argument('--patience', type=int,
                        default=50)
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
    parser.add_argument('--visualize', choices=["gates",
                                                "states", "gradients",
                                                "presoft", "matrices",
                                                "gradients_flow_pie",
                                                "trained_singular_values",
                                                "jacobian", "generate"],
                        default=None)
    parser.add_argument('--visualize_length', type=int,
                        default=300)
    parser.add_argument('--visualize_cells', action="store_true",
                        default=False)
    parser.add_argument('--local', action="store_true",
                        default=False)
    parser.add_argument('--hide_all_except', type=int,
                        default=None)

    args = parser.parse_args()

    # Print all the arguments
    logger.info("\n" + "#" * 40)
    for arg in vars(args):
        logger.info('\"' + str(arg) + '\" \t: ' + str(vars(args)[arg]))
    logger.info("#" * 40 + "\n")

    # Layers are embeded inside clockwork rnn
    assert ((args.rnn_type == 'clockwork2') and
            args.layers == 1)

    return args


def carry_hidden_state(updates, mini_batch_size, reset=False):
    state_vars = [theano.shared(
        numpy.zeros((mini_batch_size, v.shape[1].eval()),
                    dtype=numpy.float32),
        v.name + '-gen') for v, _ in updates]
    givens = [(v, x) for (v, _), x in zip(updates, state_vars)]

    # Keep the shared_variables constant
    if reset:
        f_updates = [(x, x) for x in state_vars]
    # Update the shared_variables
    else:
        f_updates = [(x, upd) for x, (_, upd) in zip(state_vars, updates)]

    return givens, f_updates
