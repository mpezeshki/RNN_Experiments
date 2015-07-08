import logging

import numpy as np

import theano

from blocks.graph import ComputationGraph
from blocks.extensions import SimpleExtension

import matplotlib
import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


class VisualizeGradients(SimpleExtension):

    def __init__(self, cost, updates, args, ploting_path=None,
                 **kwargs):
        kwargs.setdefault("after_batch", 1)
        self.text_length = 75
        self.dataset = args.dataset
        self.args = args
        self.see_several_states = (args.layers > 1 and (args.skip_connections or args.skip_output)) or (args.rnn_type in ["soft", "hard", "clockwork"])
        super(VisualizeGradients, self).__init__(**kwargs)

        outputs = [
            var for var in ComputationGraph(cost).variables if var.name == "hidden_state"]

        assert len(outputs) == 1

        if self.see_several_states:
            h = []
            dim = args.state_dim
            for i in range(args.layers):
                h.append(
                    outputs[0][:, :, dim * i: dim * (i + 1)])
        else:
            h = outputs

        cg = ComputationGraph(h)

        assert(len(cg.inputs) == 1)
        assert(cg.inputs[0].name == "features")

        state_vars = [theano.shared(
            v[0:1, :].zeros_like().eval(), v.name + '-gen')
            for v, _ in updates]
        givens = [(v, x) for (v, _), x in zip(updates, state_vars)]
        f_updates = [(x, upd) for x, (_, upd) in zip(state_vars, updates)]
        self.generate = theano.function(inputs=cg.inputs, outputs=h,
                                        givens=givens, updates=f_updates)

    def do(self, *args):
        init_ = next(self.main_loop.epoch_iterator)["features"][
            0: self.text_length, 0:1]

        hidden_state = self.generate(init_)

        layers = len(hidden_state)
        time = hidden_state[0].shape[0]

        for i in range(layers):
            plt.subplot(layers, 1, i + 1)
            for j in range(self.args.state_dim):
                plt.plot(np.arange(time), hidden_state[i][:, 0, j])
            plt.xticks(range(self.text_length), tuple(init_[:, 0]))
            plt.grid(True)
            plt.title("hidden_state_of_layer_" + str(i))
        plt.show()
