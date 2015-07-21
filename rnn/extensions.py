import logging

import numpy as np
from numpy.random import random_sample

from scipy.linalg import svd

import theano

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.serialization import secure_dump
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension

import matplotlib.pyplot as plt
from matplotlib.table import Table

from rnn.datasets.dataset import get_character, conv_into_char, get_vocab_size
from rnn.utils import carry_hidden_state

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


# Credits to Cesar Laurent
class EarlyStopping(SimpleExtension):

    """Check if a log quantity has the minimum/maximum value so far,
    and early stops the experiment if the quantity has not been better
    since `patience` number of epochs. It also saves the best best model
    so far.

    Parameters
    ----------
    record_name : str
        The name of the record to track.
    patience : int
        The number of epochs to wait before early stopping.
    path : str
        The path where to save the best model.
    notification_name : str, optional
        The name for the record to be made in the log when the current
        value of the tracked quantity is the best so far. It not given,
        'record_name' plus "best_so_far" suffix is used.
    choose_best : callable, optional
        A function that takes the current value and the best so far
        and return the best of two. By default :func:`min`, which
        corresponds to tracking the minimum value.

    Attributes
    ----------
    best_name : str
        The name of the status record to keep the best value so far.
    notification_name : str
        The name used for the notification

    """

    def __init__(self, record_name, patience, path, notification_name=None,
                 choose_best=min, **kwargs):
        self.record_name = record_name
        if not notification_name:
            notification_name = record_name + "_best_so_far"
        self.notification_name = notification_name
        self.best_name = "best_" + record_name
        self.choose_best = choose_best
        self.counter = 0
        self.path = path
        self.patience = patience
        kwargs.setdefault("after_epoch", True)
        super(EarlyStopping, self).__init__(**kwargs)

    def _dump(self):
        try:
            path = self.path + '/best'
            self.main_loop.log.current_row['saved_best_to'] = path
            logger.info("Saving log ...")
            f = open(self.path + '/log.txt', 'w')
            f.write(str(self.main_loop.log))
            f.close()
            logger.info("Dumping best model ...")
            secure_dump(
                self.main_loop.model.parameters, path, use_cpickle=True)
        except Exception:
            self.main_loop.log.current_row['saved_best_to'] = None
            raise

    def do(self, which_callback, *args):
        current_value = self.main_loop.log.current_row.get(self.record_name)
        if current_value is None:
            self.counter += 1
            return
        best_value = self.main_loop.status.get(self.best_name, None)
        if (best_value is None or
                (current_value != best_value and
                 self.choose_best(current_value, best_value) ==
                 current_value)):
            self.main_loop.status[self.best_name] = current_value
            self.main_loop.log.current_row[self.notification_name] = True
            self.counter = 0
            self._dump()
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.main_loop.log.current_row['training_finish_requested'] = True
        self.main_loop.log.current_row['patience'] = self.counter


# Credits to Alex Auvolat
class ResetStates(SimpleExtension):

    def __init__(self, state_vars, **kwargs):
        super(ResetStates, self).__init__(**kwargs)

        self.f = theano.function(
            inputs=[], outputs=[],
            updates=[(v, v.zeros_like()) for v in state_vars])

    def do(self, which_callback, *args):
        self.f()


class InteractiveMode(SimpleExtension):

    def __init__(self, **kwargs):
        kwargs.setdefault("before_training", True)
        kwargs.setdefault("on_interrupt", True)
        super(InteractiveMode, self).__init__(**kwargs)

    def do(self, *args):
        import ipdb
        ipdb.set_trace()


class SvdExtension(SimpleExtension, MonitoringExtension):

    def __init__(self, **kwargs):
        super(SvdExtension, self).__init__(**kwargs)

    def do(self, *args):
        for network in self.main_loop.model.top_bricks[-1].networks:
            w_svd = svd(network.children[0].W.get_value())
            self.main_loop.log.current_row['last_layer_W_svd' +
                                           network.name] = w_svd[1]


class TextGenerationExtension(SimpleExtension):

    def __init__(self, cost, generation_length, dataset,
                 initial_text_length, softmax_sampling,
                 updates, ploting_path=None,
                 interactive_mode=False, **kwargs):
        self.generation_length = generation_length
        self.init_length = initial_text_length
        self.dataset = dataset
        self.vocab_size = get_vocab_size(dataset)
        self.ploting_path = ploting_path
        self.softmax_sampling = softmax_sampling
        self.interactive_mode = interactive_mode
        super(TextGenerationExtension, self).__init__(**kwargs)

        # Get presoft and its computation graph
        filter_presoft = VariableFilter(theano_name="presoft")
        presoft = filter_presoft(ComputationGraph(cost).variables)
        cg = ComputationGraph(presoft)

        # Handle the theano shared variables that allow carrying the hidden
        # state
        givens, f_updates = carry_hidden_state(updates, 1)

        # Compile the theano function
        self.generate = theano.function(inputs=cg.inputs, outputs=presoft,
                                        givens=givens, updates=f_updates)

    def do(self, *args):

        # init is TIME X 1
        # This is because in interactive mode,
        # self.main_loop.epoch_iterator is not accessible.
        if self.interactive_mode:
            # TEMPORARY HACK
            iterator = self.main_loop.data_stream.get_epoch_iterator()
            init_ = next(iterator)[0][0: self.init_length, 2:3]
        else:
            iterator = self.main_loop.epoch_iterator
            init_ = next(iterator)["features"][0: self.init_length, 0:1]

        # Time X Batch X Features
        probability_array = np.zeros((0, 1, self.vocab_size))
        generated_text = init_

        logger.info("\nGeneration:")
        for i in range(self.generation_length):
            presoft = self.generate(generated_text)[0]
            # Get the last value of presoft
            last_presoft = presoft[-1:, :, :]

            # Compute the probability distribution
            probabilities = softmax(last_presoft)
            # Store it in the list
            probability_array = np.vstack([probabilities, probabilities])

            # Sample a character out of the probability distribution
            argmax = (self.softmax_sampling == 'argmax')
            last_output_sample = sample(probabilities, argmax)

            # Concatenate the new value to the text
            generated_text = np.vstack([generated_text, last_output_sample])

        # Convert with real characters
        whole_sentence = conv_into_char(generated_text[:, 0], self.dataset)
        initial_sentence = whole_sentence[:init_.shape[0]]

        logger.info(initial_sentence + '...')
        logger.info(whole_sentence)

        if self.ploting_path is not None:
            probability_plot(probability_array, initial_sentence,
                             get_character(self.dataset), self.ploting_path)

    def interactive_generate(self, initial_text, generation_length, *args):
        vocab = get_character(self.dataset)
        initial_code = []
        for char in initial_text:
            initial_code += [np.where(vocab == char)[0]]
        initial_code = np.array(initial_code)
        inputs_ = initial_code
        all_output_probabilities = []
        logger.info("\nGeneration:")
        for i in range(generation_length):
            # time x batch x features (1 x 1 x vocab_size)
            last_output = self.generate(inputs_)[-1][-1:, :, :]
            # time x features (1 x vocab_size) '0' is for removing one dim
            last_output_probabilities = softmax(last_output[0])
            all_output_probabilities += [last_output_probabilities]
            # 1 x 1
            if self.softmax_sampling == 'argmax':
                argmax = True
            else:
                argmax = False
            last_output_sample = sample(last_output_probabilities, argmax)
            inputs_ = np.vstack([inputs_, last_output_sample])
        # time x batch
        whole_sentence_code = inputs_
        # whole_sentence
        whole_sentence = ''
        for char in vocab[whole_sentence_code[:, 0]]:
            whole_sentence += char
        logger.info(whole_sentence[:initial_code.shape[0]] + ' ...')
        logger.info(whole_sentence)


# python softmax
def softmax(w):
    e = np.exp(w)
    dist = e / np.sum(e, axis=1)
    return dist


# python sampling
def sample(probs, argmax=False):
    assert(probs.shape[0] == 1)
    if argmax:
        return np.argmax(probs, axis=1)
    bins = np.add.accumulate(probs[0])
    return np.digitize(random_sample(1), bins)


# python plotting
def probability_plot(probabilities, selected, vocab, ploting_path,
                     top_n_probabilities=20, max_length=120):

    selected = selected[:max_length]
    probabilities = probabilities[:max_length]
    # target = ['a', 'b', 'c', 'd', 'e', 'f', 'a', 'b', 'c', 'd']
    # probabilities = np.random.uniform(low=0, high=1, size=(10, 6))  # T x C
    sorted_probabilities = np.zeros(probabilities.shape)
    sorted_indices = np.zeros(probabilities.shape)
    for i in range(probabilities.shape[0]):
        sorted_probabilities[i, :] = np.sort(probabilities[i, :])
        sorted_indices[i, :] = np.argsort(probabilities[i, :])
    concatenated = np.zeros((
        probabilities.shape[0], probabilities.shape[1], 2))
    concatenated[:, :, 0] = sorted_probabilities[:, ::-1]
    concatenated[:, :, 1] = sorted_indices[:, ::-1]

    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    ncols = concatenated.shape[0]
    width, height = 1.0 / (ncols + 1), 1.0 / (top_n_probabilities + 1)

    for (i, j), v in np.ndenumerate(concatenated[:, :top_n_probabilities, 0]):
        tb.add_cell(j + 1, i, height, width,
                    text=unicode(str(vocab[concatenated[i, j, 1].astype('int')]),
                                 errors='ignore'),
                    loc='center', facecolor=(1,
                                             1 - concatenated[i, j, 0],
                                             1 - concatenated[i, j, 0]))
    for i, char in enumerate(selected):
        tb.add_cell(0, i, height, width,
                    text=unicode(char, errors='ignore'),
                    loc='center', facecolor='green')
    ax.add_table(tb)

    plt.savefig(ploting_path)
