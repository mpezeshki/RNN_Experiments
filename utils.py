import theano
import argparse
import numpy as np
from blocks.algorithms import (StepClipping, CompositeRule,
                               RMSProp, Momentum, Adam)
floatX = theano.config.floatX


def parse_args():
    parser = argparse.ArgumentParser(description='RNN experiment')
    parser.add_argument('--save_path', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--load_path', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--input_dim', type=int, default=None)
    parser.add_argument('--state_dim', type=int, default=None)
    parser.add_argument('--target_dim', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--clipping', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--algorithm',
                        choices=['rms_prop', 'adam', 'sgd'],
                        default='sgd')
    parser.add_argument('rnn_type', choices=['lstm', 'simple'], default='lstm')
    return parser.parse_args()


def learning_algorithm(args):
    name = args.algorithm
    learning_rate = float(args.learning_rate)
    momentum = args.momentum
    clipping_threshold = args.clipping
    if name == 'adam':
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        adam = Adam(learning_rate=learning_rate)
        step_rule = CompositeRule([adam, clipping])
    elif name == 'rms_prop':
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        rms_prop = RMSProp(learning_rate=learning_rate)
        step_rule = CompositeRule([clipping, rms_prop])
    else:
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        sgd_momentum = Momentum(learning_rate=learning_rate, momentum=momentum)
        step_rule = CompositeRule([clipping, sgd_momentum])
    return step_rule
