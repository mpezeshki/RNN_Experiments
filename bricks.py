from theano import tensor
from theano.ifelse import ifelse

from blocks.bricks import Initializable, MLP, Tanh
from blocks.bricks.base import application, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.initialization import IsotropicGaussian, Constant
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
        W = shared_floatx_nans((self.length, self.dim), name='W')
        self.params.append(W)
        add_role(W, WEIGHT)
        b = shared_floatx_nans((self.dim,), name='W')
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

        return next_states, time + 1

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.params[1][None, :], batch_size, 0),
                self.params[2][None, :]]


class SoftGatedRecurrent(BaseRecurrent, Initializable):

    """The traditional recurrent transition.
    The most well-known recurrent transition: a matrix multiplication,
    optionally followed by a non-linearity.
    Parameters
    ----------
    dim : int
        The dimension of the hidden state
    activation : :class:`.Brick`
        The brick to apply as activation.
    Notes
    -----
    See :class:`.Initializable` for initialization parameters.
    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, dim_prev_layer, activation, **kwargs):
        super(SoftGatedRecurrent, self).__init__(**kwargs)
        self.dim = dim
        self.gate_activation = Tanh()
        self.children = [activation, self.gate_activation]
        self.dim_prev_layer = dim_prev_layer

    @property
    def W(self):
        return self.params[0]

    @property
    def W_g(self):
        return self.params[2]

    @property
    def b_g(self):
        return self.params[3]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (SoftGatedRecurrent.apply.sequences +
                    SoftGatedRecurrent.apply.states):
            return self.dim
        return super(SoftGatedRecurrent, self).get_dim(name)

    def _allocate(self):
        self.params.append(shared_floatx_nans((self.dim, self.dim), name="W"))
        add_role(self.params[0], WEIGHT)
        self.params.append(shared_floatx_zeros((self.dim,),
                                               name="initial_state"))
        add_role(self.params[1], INITIAL_STATE)

        self.params.append(
            shared_floatx_nans((self.dim + self.dim_prev_layer, self.dim),
                               name="W_g"))
        add_role(self.params[2], WEIGHT)

        self.params.append(shared_floatx_nans((self.dim,), name="b_g"))
        add_role(self.params[3], BIAS)

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)
        self.weights_init.initialize(self.W_g, self.rng)
        self.biases_init.initialize(self.b_g, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=[])
    def apply(self, inputs=None, states=None, mask=None):
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
        """

        # Compute the gate value
        gate_input = tensor.concatenate((inputs, states), axis=1)
        gate_value = self.gate_activation.apply(
            tensor.dot(gate_input, self.W_g) + self.b_g)

        next_states = inputs + tensor.dot(states, self.W)
        next_states = self.children[0].apply(next_states)

        # Apply the gating
        next_states = gate_value * next_states + (1 - gate_value) * states

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return tensor.repeat(self.params[1][None, :], batch_size, 0)
