import logging

import numpy as np

import theano
from theano import tensor
from theano.compile import Mode

from blocks.graph import ComputationGraph
from rnn.datasets.dataset import get_character

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_gradients_flow_pie(hidden_states, updates,
                                 args, text='[done]. Finally'):
    unfolding_length = len(text)
    variables = ComputationGraph(hidden_states).variables

    if args.rnn_type == 'lstm':
        rnn_type = 'lstm'
    elif args.rnn_type == 'simple':
        rnn_type = 'simplerecurrent'
    else:
        raise NotImplemented
    states = []
    for d in range(args.layers):
        states.append([variable for variable in variables
                       if variable.name == (rnn_type + '_' +
                                            str(d) + '_apply_states')][1])
        # [1] is because there are two '*_apply_states' in variables.
    pre_rnns = [variable for variable in variables
                if ((variable.name is not None) and
                    ('pre_rnn' in variable.name))]

    grads = []
    for i in range(unfolding_length):
        grads.append(tensor.sum(tensor.abs_(tensor.grad(
            tensor.mean(tensor.abs_(pre_rnns[0][i])),
            pre_rnns[0:1])), axis=0))

        for layer, state in enumerate(states):
            grads.append(tensor.sum(tensor.abs_(tensor.grad(
                tensor.mean(tensor.abs_(state[i])),
                pre_rnns[0:layer + 1])), axis=0))

    # Handle the theano shared variables for the state
    state_vars = [theano.shared(
        v[0:1, :].zeros_like().eval(), v.name + '-gen')
        for v, _ in updates]
    givens = [(v, x) for (v, _), x in zip(updates, state_vars)]
    f_updates = [(x, upd) for x, (_, upd) in zip(state_vars, updates)]

    # Compile the function
    logger.info("The compilation of the function has started")
    compiled_functions = [theano.function(
        inputs=ComputationGraph(grad).inputs,
        outputs=grad,
        givens=givens, updates=f_updates,
        mode=Mode(optimizer=None)) for grad in grads]
    logger.info("The function has been compiled")

    # input text
    vocab = get_character(args.dataset)
    code = []
    for char in text:
        code += [np.where(vocab == char)[0]]
    code = np.array(code)

    res = [f(code) for f in compiled_functions]
    all_time_steps = []
    for i in range(unfolding_length):
        temp = []
        for d in range(args.layers + 1):
            temp.append(np.sum(np.abs(res[i * (args.layers + 1) + d]),
                               axis=(1, 2)))
        all_values = np.vstack([layer / np.sum(layer, axis=0)
                                for layer in temp])
        all_time_steps += [all_values.T[:, ::-1]]
    # +1 is to show inputs as well
    plot_pie_charts(data=all_time_steps, layers=args.layers + 1,
                    time_steps=unfolding_length,
                    path=args.save_path + '/pie.png')


def plot_pie_charts(data, layers, time_steps, path, per_layer=1):
    # These are the "Table 20" colors as RGB.
    table = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(table)):
        r, g, b = table[i]
        table[i] = (r / 255., g / 255., b / 255.)

    plt.figure(figsize=(time_steps, per_layer * layers))
    gs1 = gridspec.GridSpec(per_layer * layers, time_steps)
    gs1.update(wspace=0.025, hspace=0.05)

    for i in range(per_layer * layers * time_steps):
        ax1 = plt.subplot(gs1[i])
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')

        sizes = data[i % time_steps][
            :, (i / time_steps) % (per_layer * layers)]
        colors = table
        wedges, _ = ax1.pie(x=sizes, colors=colors,
                            startangle=90, labeldistance=0.8)
        for w in wedges:
            w.set_linewidth(0)

    plt.savefig(path)
