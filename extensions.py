# Credits to Cesar Laurent
from blocks.serialization import secure_dump
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from scipy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


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
            self.main_loop.log.current_row['saved_best_to'] = self.path
            secure_dump(self.main_loop, self.path)
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


class SvdExtension(SimpleExtension, MonitoringExtension):
    def __init__(self, **kwargs):
        super(SvdExtension, self).__init__(**kwargs)

    def do(self, *args):
        for network in self.main_loop.model.top_bricks[-1].networks:
            w_svd = svd(network.children[0].W.get_value())
            self.main_loop.log.current_row['last_layer_W_svd' +
                                           network.name] = w_svd[1]


def probability_plot(prob, alphabet_list, target, top_n_probabilities=5):
    # target = ['a', 'b', 'c', 'd', 'e', 'f', 'a', 'b', 'c', 'd']
    # prob = np.random.uniform(low=0, high=1, size=(10, 6))  # T x C
    sorted_prob = np.zeros(prob.shape)
    sorted_indices = np.zeros(prob.shape)
    for i in range(prob.shape[0]):
        sorted_prob[i, :] = np.sort(prob[i, :])
        sorted_indices[i, :] = np.argsort(prob[i, :])
    concatenated = np.zeros((prob.shape[0], prob.shape[1], 2))
    concatenated[:, :, 0] = sorted_prob[:, ::-1]
    concatenated[:, :, 1] = sorted_indices[:, ::-1]

    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    ncols = concatenated.shape[0]
    width, height = 1.0 / (ncols + 1), 1.0 / (top_n_probabilities + 1)

    for (i, j), v in np.ndenumerate(concatenated[:, :top_n_probabilities, 0]):
        tb.add_cell(j + 1, i, height, width,
                    text=alphabet_list[concatenated[i, j, 1].astype('int')],
                    loc='center', facecolor=(1,
                                             1 - concatenated[i, j, 0],
                                             1 - concatenated[i, j, 0]))
    for i, char in enumerate(target):
        tb.add_cell(0, i, height, width,
                    text=char,
                    loc='center', facecolor='green')
    ax.add_table(tb)

    plt.show()
