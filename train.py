import logging

from blocks.algorithm import GradientDescent
from blocks.extensions import Printing
from blocks.extensions.monitoring import (
    TrainingDataMonitoring, DataStreamMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model

from utils import learning_algorithm

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


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
            Printing(),
            Checkpoint(args.save_path, after_epoch=True)
        ]
    )
    main_loop.run()
