from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

from blocks.bricks import Initializable, Tanh, Activation
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
        return self.parameters[0]

    @property
    def b(self):
        return self.parameters[1]

    def _allocate(self):
        W = shared_floatx_nans((self.length, self.dim), name='W_lookup')
        self.parameters.append(W)
        add_role(W, WEIGHT)
        b = shared_floatx_nans((self.dim,), name='b_lookup')
        self.parameters.append(b)
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
        return self.parameters[0]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (ClockworkBase.apply.sequences +
                    ClockworkBase.apply.states):
            return self.dim
        return super(ClockworkBase, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim), name="W"))
        add_role(self.parameters[0], WEIGHT)
        self.parameters.append(shared_floatx_zeros((self.dim,),
                                               name="initial_state"))
        add_role(self.parameters[1], INITIAL_STATE)

        self.parameters.append(shared_floatx_zeros((1,), name="initial_time"))
        add_role(self.parameters[2], INITIAL_STATE)

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
        return [tensor.repeat(self.parameters[1][None, :], batch_size, 0),
                self.parameters[2][None, :]]


class SoftGatedRecurrent(BaseRecurrent, Initializable):

    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, mlp=None,
                 **kwargs):
        super(SoftGatedRecurrent, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        self.activation = activation

        # The activation of the mlp should be a Logistic function
        self.mlp = mlp

        self.children = [activation, mlp]

    @property
    def state_to_state(self):
        return self.parameters[0]

    @property
    def matrix_gate(self):
        return self.parameters[1]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in ['inputs', 'states']:
            return self.dim
        return super(SoftGatedRecurrent, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                              name='state_to_state'))
        self.parameters.append(shared_floatx_zeros((self.dim,),
                                               name="initial_state"))
        add_role(self.parameters[0], WEIGHT)
        add_role(self.parameters[1], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)

    @recurrent(sequences=['mask', 'inputs'], states=['states'],
               outputs=['states', "gate_value"], contexts=[])
    def apply(self, inputs, states, mask=None):
        """Apply the gated recurrent transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim)
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.
        """
        # Concatenate the inputs of the MLP
        mlp_input = tensor.concatenate((inputs, states), axis=1)

        # Compute the output of the MLP
        gate_value = self.mlp.apply(mlp_input)

        # TODO: Find a way to remove the following "hack".
        # Simply removing the two next lines won't work
        gate_value = gate_value[:, 0]
        gate_value = gate_value[:, None]

        # Compute the next_states value, before gating
        next_states = self.activation.apply(
            states.dot(self.state_to_state) + inputs)

        # Apply the gating
        next_states = (next_states * gate_value +
                       states * (1 - gate_value))

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states, gate_value

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.parameters[2][None, :], batch_size, 0)]


class HardGatedRecurrent(BaseRecurrent, Initializable):

    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, mlp=None,
                 **kwargs):
        super(HardGatedRecurrent, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        self.activation = activation

        # The activation of the mlp should be a Logistic function
        self.mlp = mlp

        # The random stream
        self.randomstream = MRG_RandomStreams()

        self.children = [activation, mlp]

    @property
    def state_to_state(self):
        return self.parameters[0]

    @property
    def matrix_gate(self):
        return self.parameters[1]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in ['inputs', 'states']:
            return self.dim
        return super(HardGatedRecurrent, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                              name='state_to_state'))
        self.parameters.append(shared_floatx_zeros((self.dim,),
                                               name="initial_state"))
        add_role(self.parameters[0], WEIGHT)
        add_role(self.parameters[1], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)

    @recurrent(sequences=['mask', 'inputs'],
               states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, states, mask=None):
        """Apply the gated recurrent transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim)
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.
        """
        # Concatenate the inputs of the MLP
        mlp_input = tensor.concatenate((inputs, states), axis=1)

        # Compute the output of the MLP
        gate_value = self.mlp.apply(mlp_input)
        random = self.randomstream.uniform((1,))

        # TODO: Find a way to remove the following "hack".
        # Simply removing the two next lines won't work
        gate_value = gate_value[:, 0]
        gate_value = gate_value[:, None]

        # Compute the next_states value, before gating
        next_states = self.activation.apply(
            states.dot(self.state_to_state) + inputs)

        # Apply the gating
        next_states = tensor.switch(tensor.le(random[0], gate_value),
                                    next_states,
                                    states)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.parameters[2][None, :], batch_size, 0)]


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

        self.parameters = [
            self.W_state, self.initial_state_, self.initial_cells]

    def _initialize(self):
        self.weights_init.initialize(self.parameters[0], self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells', 'in_gate',
                                     'forget_gate', 'out_gate'])
    def apply(self, inputs, states, cells, mask=None):
        def slice_last(x, no):
            return x[:, no * self.dim: (no + 1) * self.dim]

        nonlinearity = self.children[0].apply

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0))
        forget_gate = tensor.nnet.sigmoid(slice_last(activation, 1))
        next_cells = (forget_gate * cells +
                      in_gate * nonlinearity(slice_last(activation, 3)))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 2))
        next_states = out_gate * nonlinearity(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)

        return next_states, next_cells, in_gate, forget_gate, out_gate

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :], batch_size, 0)]


class HardLogistic(Activation):

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.hard_sigmoid(input_)
