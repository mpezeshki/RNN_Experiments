import logging

import numpy as np

import theano

from blocks.algorithms import (Adam, CompositeRule, GradientDescent,
                               Momentum, RMSProp, StepClipping)
from blocks.extensions import Printing
from blocks.extensions.monitoring import (
    TrainingDataMonitoring, DataStreamMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from extensions import EarlyStopping

floatX = theano.config.floatX
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


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


def train_model(cost, cross_entropy, train_stream, valid_stream, args):

    # Define the model
    model = Model(cost)

    step_rule = learning_algorithm(args)
    cg = ComputationGraph(cost)
    algorithm = GradientDescent(cost=cost, step_rule=step_rule,
                                params=cg.parameters)

    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            TrainingDataMonitoring([cost], prefix='train'),
            DataStreamMonitoring([cost, cross_entropy],
                                 valid_stream, prefix='valid'),
            Checkpoint(args.save_path, after_epoch=True),
            EarlyStopping(cost, args.patience, args.save_path),
            Printing()
        ]
    )
    main_loop.run()
