import numpy as np
import theano
from blocks.algorithms import (Adam, CompositeRule,
                               Momentum, RMSProp, StepClipping,
                               RemoveNotFinite)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

floatX = theano.config.floatX


def learning_algorithm(learning_rate, momentum=0.0,
                       clipping_threshold=100, algorithm='sgd'):
    if algorithm == 'adam':
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        adam = Adam(learning_rate=learning_rate)
        # [adam, clipping] means 'step clipping'
        # [clipping, adam] means 'gradient clipping'
        step_rule = CompositeRule([adam, clipping])
    elif algorithm == 'rms_prop':
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        rms_prop = RMSProp(learning_rate=learning_rate)
        rm_non_finite = RemoveNotFinite()
        step_rule = CompositeRule([clipping, rms_prop, rm_non_finite])
    elif algorithm == 'sgd':
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        sgd_momentum = Momentum(learning_rate=learning_rate, momentum=momentum)
        rm_non_finite = RemoveNotFinite()
        step_rule = CompositeRule([clipping, sgd_momentum, rm_non_finite])
    else:
        raise NotImplementedError
    return step_rule


def plot_signals(input_seq, target_seq, output_seq):
    # Shapes --> [num_time_steps x batch_size x dim]
    # We just plot one of the sequences
    plt.close('all')
    plt.figure()

    # Graph 1
    ax1 = plt.subplot(211)
    plt.plot(input_seq[:, 0, :])
    plt.grid()
    ax1.set_title('Input sequence')

    # Graph 2
    ax2 = plt.subplot(212)
    true_targets = plt.plot(target_seq[:, 0, :])

    guessed_targets = plt.plot(output_seq[:, 0, :], linestyle='--')
    plt.grid()
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')

    # Save as a file
    plt.savefig('RNN_seq.png')
    print("Figure is saved as a .png file.")
