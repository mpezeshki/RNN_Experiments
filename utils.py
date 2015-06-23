import argparse
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='RNN_experiment')
    parser.add_argument('--mini_batch_size', type=int, default=15)
    parser.add_argument('--time_length', type=int, default=150)
    parser.add_argument('--context', type=int, default=1)
    parser.add_argument('--tot_num_char', type=int, default=None)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_path', type=str,
                        default="/data/lisatmp3/zablocki")
    # default="/media/win/Users/Eloi/tmp")
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--state_dim', type=int, default=10)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--skip_connections', action='store_true',
                        default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--clipping', type=float, default=10)
    parser.add_argument('--algorithm',
                        choices=['rms_prop', 'adam', 'sgd'],
                        default='adam')
    parser.add_argument('--rnn_type', choices=['lstm', 'simple', 'clockwork',
                                               'soft', 'hard'],
                        default='simple')
    parser.add_argument('--dataset', choices=['wikipedia', 'penntree'],
                        default='penntree')
    parser.add_argument('--monitoring_freq', type=int, default=1000)
    args = parser.parse_args()

    logger.info("\n" + "#" * 40)
    for arg in vars(args):
        logger.info('\"' + str(arg) + '\" \t: ' + str(vars(args)[arg]))
    logger.info("#" * 40 + "\n")
    return args
