import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='RNN_experiment')
    parser.add_argument('--mini_batch_size', type=int, default=10)
    parser.add_argument('--time_length', type=int, default=150)
    parser.add_argument('--context', type=int, default=20)
    parser.add_argument('--load_path', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--save_path', type=str,
                        # default="/data/lisatmp3/zablocki")
                        default="/media/win/Users/Eloi/tmp")
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--state_dim', type=int, default=100)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--skip_connections', choices=[True, False],
                        default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--clipping', type=float, default=10)
    parser.add_argument('--algorithm',
                        choices=['rms_prop', 'adam', 'sgd'],
                        default='adam')
    parser.add_argument('--rnn_type', choices=['lstm', 'simple', 'clockwork'],
                        default='lstm')
    parser.add_argument('--dataset', choices=['wikipedia', 'penntree'],
                        default='penntree')

    return parser.parse_args()
