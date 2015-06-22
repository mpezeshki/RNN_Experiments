import numpy

from theano import tensor

from blocks.bricks import Initializable, Tanh, Logistic
from blocks.bricks.base import application, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent
# from blocks.initialization import IsotropicGaussian, Constant
from blocks.roles import add_role, WEIGHT, BIAS, INITIAL_STATE
from blocks.utils import (
    check_theano_variable, shared_floatx_nans, shared_floatx_zeros)


class LookupTable(Initializable):

    """Encapsulates representations of a range of integers.
    Parameters
    ----------
    length : int
        The size of the lookup table, or in other words, one plus the
        maximum index for which a representation is contained.
    dim : int
        The dimensionality of representations.
    Notes
    -----
    See :class:`.Initializable` for initialization parameters.
    """
    has_bias = True

    @lazy(allocation=['length', 'dim'])
    def __init__(self, length, dim, **kwargs):
        super(LookupTable, self).__init__(**kwargs)
        self.length = length
        self.dim = dim

    @property
    def W(self):
        return self.params[0]

    @property
    def b(self):
        return self.params[1]

    def _allocate(self):
        W = shared_floatx_nans((self.length, self.dim), name='W_lookup')
        self.params.append(W)
        add_role(W, WEIGHT)
        b = shared_floatx_nans((self.dim,), name='b_lookup')
        self.params.append(b)
        add_role(b, BIAS)

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)
        self.biases_init.initialize(self.b, self.rng)

    @application
    def apply(self, indices):
        """Perform lookup.
        Parameters
        ----------
        indices : :class:`~tensor.TensorVariable`
            The indices of interest. The dtype must be integer.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Representations for the indices of the query. Has :math:`k+1`
            dimensions, where :math:`k` is the number of dimensions of the
            `indices` parameter. The last dimension stands for the
            representation element.
        """
        check_theano_variable(indices, None, "int")
        output_shape = [indices.shape[i]
                        for i in range(indices.ndim)] + [self.dim]
        return self.W[indices.flatten()].reshape(output_shape) + self.b


# Very similar to the SimpleRecurrent implementation. But the computation is
# made one every `period` time steps. This brick carries the time as a state
class ClockworkBase(BaseRecurrent, Initializable):

    @lazy(allocation=['dim'])
    def __init__(self, dim, period, activation, **kwargs):
        super(ClockworkBase, self).__init__(**kwargs)
        self.dim = dim
        self.period = period
        self.children = [activation]

    @property
    def W(self):
        return self.params[0]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (ClockworkBase.apply.sequences +
                    ClockworkBase.apply.states):
            return self.dim
        return super(ClockworkBase, self).get_dim(name)

    def _allocate(self):
        self.params.append(shared_floatx_nans((self.dim, self.dim), name="W"))
        add_role(self.params[0], WEIGHT)
        self.params.append(shared_floatx_zeros((self.dim,),
                                               name="initial_state"))
        add_role(self.params[1], INITIAL_STATE)

        self.params.append(shared_floatx_zeros((1,), name="initial_time"))
        add_role(self.params[2], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'time'],
               outputs=['states', 'time'], contexts=[])
    def apply(self, inputs=None, states=None, time=None, mask=None):
        """Apply the simple transition.
        Parameters
        ----------
        inputs : :class:`~tensor.TensorVariable`
            The 2D inputs, in the shape (batch, features).
        states : :class:`~tensor.TensorVariable`
            The 2D states, in the shape (batch, features).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if
            there is data available, 0 if not. Assumed to be 1-s
            only if not given.
        time : :class:`~tensor.TensorVariable`
            A number representing the time steps currently computed
        """

        # TODO check which one is faster: switch or ifelse
        next_states = tensor.switch(tensor.eq(time[0, 0] % self.period, 0),
                                    self.children[0].apply(
            inputs + tensor.dot(states, self.W)),
            states)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)

        time = time + tensor.ones_like(time)
        return next_states, time

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.params[1][None, :], batch_size, 0),
                self.params[2][None, :]]


class SoftGatedRecurrent(BaseRecurrent, Initializable):

    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        super(SoftGatedRecurrent, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        self.children = [activation, gate_activation]

    @property
    def state_to_state(self):
        return self.params[0]

    @property
    def matrix_gate(self):
        return self.params[1]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in ['inputs', 'states', 'gate_inputs']:
            return self.dim
        return super(SoftGatedRecurrent, self).get_dim(name)

    def _allocate(self):
        self.params.append(shared_floatx_nans((self.dim, self.dim),
                           name='state_to_state'))
        self.params.append(shared_floatx_nans((2 * self.dim, 1),
                           name='matrix_gate'))
        self.params.append(shared_floatx_zeros((self.dim,),
                           name="initial_state"))
        for i in range(2):
            if self.params[i]:
                add_role(self.params[i], WEIGHT)
        add_role(self.params[2], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)
        self.weights_init.initialize(self.matrix_gate, self.rng)

    @recurrent(sequences=['mask', 'inputs', 'gate_inputs'],
               states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, gate_inputs, states, mask=None):
        """Apply the gated recurrent transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim)
        gate_inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs to the gates in the
            shape (batch_size, dim).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.
        """
        a = tensor.concatenate((gate_inputs, states), axis=1)
        gate_value = self.gate_activation.apply(
            a.dot(self.matrix_gate))

        # TODO: Find a way to remove the follwing "hack".
        # Simply removing the two next lines won't work
        gate_value = gate_value[:, 0]
        gate_value = gate_value[:, None]

        next_states = self.activation.apply(
            states.dot(self.state_to_state) + inputs)

        next_states = (next_states * gate_value +
                       states * (1 - gate_value))

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.params[2][None, :], batch_size, 0)]


class LSTM(BaseRecurrent, Initializable):

    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        self.children = [activation]

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(LSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4 * self.dim),
                                          name='W_state')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim,),
                                                 name="initial_cells")
        add_role(self.W_state, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)

        self.params = [
            self.W_state, self.initial_state_, self.initial_cells]

    def _initialize(self):
        self.weights_init.initialize(self.params[0], self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        def slice_last(x, no):
            return x[:, no * self.dim: (no + 1) * self.dim]

        nonlinearity = self.children[0].apply

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0))
        forget_gate = tensor.nnet.sigmoid(slice_last(activation, 1))
        next_cells = (forget_gate * cells +
                      in_gate * nonlinearity(slice_last(activation, 2)))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 3))
        next_states = out_gate * nonlinearity(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)

        return next_states, next_cells

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :], batch_size, 0)]
