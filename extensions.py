import theano
from blocks.graph import ComputationGraph
from blocks.serialization import secure_dump
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from scipy.linalg import svd
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table
from dataset import get_character
from numpy.random import random_sample
import logging

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
            secure_dump(self.main_loop.model.params, path, use_cpickle=True)
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


class TestSave(SimpleExtension):

    def __init__(self, **kwargs):
        super(TestSave, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        print("here")


class SvdExtension(SimpleExtension, MonitoringExtension):

    def __init__(self, **kwargs):
        super(SvdExtension, self).__init__(**kwargs)

    def do(self, *args):
        for network in self.main_loop.model.top_bricks[-1].networks:
            w_svd = svd(network.children[0].W.get_value())
            self.main_loop.log.current_row['last_layer_W_svd' +
                                           network.name] = w_svd[1]


# Help from Alex Auvolat
class TextGenerationExtension(SimpleExtension):

    def __init__(self, outputs, generation_length, dataset,
                 initial_text_length, softmax_sampling,
                 updates, ploting_path=None, **kwargs):
        self.generation_length = generation_length
        self.initial_text_length = initial_text_length
        self.dataset = dataset
        self.ploting_path = ploting_path
        self.softmax_sampling = softmax_sampling
        super(TextGenerationExtension, self).__init__(**kwargs)

        # TODO: remove the commented lines when debugged
        # inputs = [variable for variable in variables
        #           if variable.name == 'features']

        cg = ComputationGraph(outputs)
        assert(len(cg.inputs) == 1)
        assert(cg.inputs[0].name == "features")

        state_vars = [theano.shared(
            v[0:1, :].zeros_like().eval(), v.name + '-gen')
            for v, _ in updates]
        givens = [(v, x) for (v, _), x in zip(updates, state_vars)]
        f_updates = [(x, upd) for x, (_, upd) in zip(state_vars, updates)]
        self.generate = theano.function(inputs=cg.inputs, outputs=outputs,
                                        givens=givens, updates=f_updates)

    def do(self, *args):

        # +1 is for one output (consider context = self.initial_text_length)
        # time x batch
        init_ = next(self.main_loop.data_stream.get_epoch_iterator(
        ))[0][2:2 + self.initial_text_length + 64, 3:4]
        inputs_ = init_
        all_output_probabilities = []
        logger.info("\nGeneration:")
        for i in range(self.generation_length):
            # time x batch x features (1 x 1 x vocab_size)
            last_output = self.generate(inputs_)[0][-1:, :, :]
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
        vocab = get_character(self.dataset)
        # whole_sentence
        whole_sentence = ''
        for char in vocab[whole_sentence_code[:, 0]]:
            whole_sentence += char
        logger.info(whole_sentence[:init_.shape[0]] + ' ...')
        logger.info(whole_sentence)

        if self.ploting_path is not None:
            all_output_probabilities_array = np.zeros(
                (self.generation_length, all_output_probabilities[0].shape[1]))
            for i, output_probabilities in enumerate(all_output_probabilities):
                all_output_probabilities_array[i] = output_probabilities
            probability_plot(all_output_probabilities_array,
                             whole_sentence[init_.shape[0]:],
                             vocab, self.ploting_path)


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
                     top_n_probabilities=20, max_length=30):
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
                    text=unicode(vocab[concatenated[i, j, 1].astype('int')],
                                 errors='ignore'),
                    loc='center', facecolor=(1,
                                             1 - concatenated[i, j, 0],
                                             1 - concatenated[i, j, 0]))
    for i, char in enumerate(selected):
        tb.add_cell(0, i, height, width,
                    text=unicode(char, errors='ignore'),
                    loc='center', facecolor='green')
    ax.add_table(tb)

    plt.savefig('self.ploting_path')
