import logging

import os

import numpy as np

import theano

from blocks.algorithms import (Adam, CompositeRule, GradientDescent,
                               Momentum, RMSProp, StepClipping,
                               RemoveNotFinite)
from blocks.extensions import Printing, ProgressBar
from blocks.extensions.monitoring import (
    TrainingDataMonitoring, DataStreamMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from extensions import EarlyStopping, ResetStates


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
        rm_non_finite = RemoveNotFinite()
        step_rule = CompositeRule([clipping, rms_prop, rm_non_finite])
    else:
        clipping = StepClipping(threshold=np.cast[floatX](clipping_threshold))
        sgd_momentum = Momentum(learning_rate=learning_rate, momentum=momentum)
        rm_non_finite = RemoveNotFinite()
        step_rule = CompositeRule([clipping, sgd_momentum, rm_non_finite])
    return step_rule


def train_model(cost, cross_entropy, train_stream, valid_stream,
                updates, args):

    # Define the model
    model = Model(cost)

    step_rule = learning_algorithm(args)
    cg = ComputationGraph(cost)
    logger.info(cg.parameters)

    algorithm = GradientDescent(cost=cost, step_rule=step_rule,
                                params=cg.parameters)
    algorithm.add_updates(updates)

    # Creating 'best' folder for saving the best model.
    best_path = os.path.join(args.save_path, 'best')
    if not os.path.exists(best_path):
        os.mkdir(best_path)
    early_stopping = EarlyStopping('valid_cross_entropy',
                                   args.patience, best_path,
                                   every_n_batches=args.monitoring_freq)
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            TrainingDataMonitoring([cost], prefix='train'),
            DataStreamMonitoring([cost, cross_entropy],
                                 valid_stream, prefix='valid',
                                 every_n_batches=args.monitoring_freq),
            Checkpoint(args.save_path, every_n_batches=args.monitoring_freq,
                       after_epoch=True),
            ResetStates([v for v, _ in updates], every_n_batches=100),
            early_stopping,
            Printing(every_n_batches=args.monitoring_freq),
            ProgressBar(),
        ]
    )
    main_loop.run()
