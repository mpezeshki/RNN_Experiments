import os
import logging

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.table import Table

import theano

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from rnn.utils import carry_hidden_state
from rnn.datasets.dataset import has_indices, conv_into_char, get_output_size

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def visualize_generate(cost, hidden_states, updates,
                       train_stream, valid_stream,
                       args):

    use_indices = has_indices(args.dataset)
    output_size = get_output_size(args.dataset)

    # Get presoft and its computation graph
    filter_presoft = VariableFilter(theano_name="presoft")
    presoft = filter_presoft(ComputationGraph(cost).variables)[0]
    cg = ComputationGraph(presoft)

    # Handle the theano shared variables that allow carrying the hidden
    # state
    givens, f_updates = carry_hidden_state(updates, 1, reset=not(use_indices))

    if args.hide_all_except is not None:
        pass

    # Compile the theano function
    compiled = theano.function(inputs=cg.inputs, outputs=presoft,
                               givens=givens, updates=f_updates)

    epoch_iterator = train_stream.get_epoch_iterator()
    for num in range(10):
        all_ = next(epoch_iterator)
        all_sequence = all_[0][:, 0:1]
        targets = all_[1][:, 0:1]

        # In the case of characters and text
        if use_indices:
            init_ = all_sequence[:args.initial_text_length]

            # Time X Features
            probability_array = np.zeros((0, output_size))
            generated_text = init_

            for i in range(args.generated_text_lenght):
                presoft = compiled(generated_text)
                # Get the last value of presoft
                last_presoft = presoft[-1:, 0, :]

                # Compute the probability distribution
                probabilities = softmax(last_presoft)
                # Store it in the list
                probability_array = np.vstack([probability_array,
                                               probabilities])

                # Sample a character out of the probability distribution
                argmax = (args.softmax_sampling == 'argmax')
                last_output_sample = sample(probabilities, argmax)[:, None, :]

                # Concatenate the new value to the text
                generated_text = np.vstack(
                    [generated_text, last_output_sample])

                ploting_path = None
                if args.save_path is not None:
                    ploting_path = os.path.join(
                        args.save_path, 'prob_plot.png')

                # Convert with real characters
                whole_sentence = conv_into_char(
                    generated_text[:, 0], args.dataset)
                initial_sentence = whole_sentence[:init_.shape[0]]
                selected_sentence = whole_sentence[init_.shape[0]:]

                logger.info(''.join(initial_sentence) + '...')
                logger.info(''.join(whole_sentence))

                if ploting_path is not None:
                    probability_plot(probability_array, selected_sentence,
                                     args.dataset, ploting_path)

        # In the case of sine wave dataset for example
        else:
            presoft = compiled(all_sequence)

            time_plot = presoft.shape[0] - 1

            plt.plot(np.arange(time_plot),
                     targets[:time_plot, 0, 0],
                     label="target")
            plt.plot(np.arange(time_plot), presoft[:time_plot, 0, 0],
                     label="predicted")
            plt.legend()
            plt.grid(True)
            if args.local:
                plt.show()
            else:
                plt.savefig((args.save_path +
                             "/visualize_generate_" + str(num) + ".png"))
            logger.info("Figure \"visualize_generate_" + str(num) +
                        ".png\" saved at directory: " + args.save_path)

# python softmax
def softmax(w):
    e = np.exp(w)
    dist = e / np.sum(e, axis=1)
    return dist


def sigmoid(w):
    return 1 / (1 + np.exp(-w))


# python sampling
def sample(probs, argmax=False):
    assert(probs.shape[0] == 1)
    if argmax:
        return np.argmax(probs, axis=1)
    bins = np.add.accumulate(probs[0])
    return np.digitize(np.random.random_sample(1), bins)


# python plotting
def probability_plot(probabilities, selected_sentence, dataset, ploting_path,
                     top_n_probabilities=20, max_length=120):

    # Pyplot options
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    ncols = probabilities.shape[0]
    width, height = 1.0 / (ncols + 1), 1.0 / (top_n_probabilities + 1)

    # Truncate the time
    selected_sentence = selected_sentence[:max_length]
    probabilities = probabilities[:max_length]

    # Sort the frequencies
    sorted_indices = np.argsort(probabilities, axis=1)
    probabilities = probabilities[
        np.repeat(np.arange(probabilities.shape[0])[
            :, None], probabilities.shape[1], axis=1),
        sorted_indices][:, ::-1]

    # Truncate the probabilities
    probabilities = probabilities[:, :top_n_probabilities]

    for (i, j), _ in np.ndenumerate(probabilities):
        tb.add_cell(j + 1, i, height, width,
                    text=unicode(str(conv_into_char(sorted_indices[i, j, 1],
                                                    dataset)[0]),
                                 errors='ignore'),
                    loc='center',
                    facecolor=(1,
                               1 - probabilities[i, j, 0],
                               1 - probabilities[i, j, 0]))

    for i, char in enumerate(selected_sentence):
        tb.add_cell(0, i, height, width,
                    text=unicode(char, errors='ignore'),
                    loc='center', facecolor='green')
    ax.add_table(tb)

    plt.savefig(ploting_path)
