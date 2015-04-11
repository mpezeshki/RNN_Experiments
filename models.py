import inspect
import logging
from functools import wraps

from picklable_itertools.extras import equizip
import theano
from theano import tensor, Variable

from blocks.bricks import Initializable
from blocks.bricks.base import Application, application, lazy
from blocks.initialization import NdarrayInitialization
from blocks.utils import (pack, shared_floatx_nans, dict_union, dict_subset,
                          is_shared_variable)
from blocks.bricks.recurrent import BaseRecurrent

logger = logging.getLogger()

unknown_scan_input = """

Your function uses a non-shared variable other than those given \
by scan explicitly. That can significantly slow down `tensor.grad` \
call. Did you forget to declare it in `contexts`?"""


def recurrent(*args, **kwargs):
    """Wraps an apply method to allow its iterative application.

    This decorator allows you to use implementation of an RNN
    transition to process sequences without writing the
    iteration-related code again and again. In the most general form
    information flow of a recurrent network can be described as
    follows: depending on the context variables and driven by input
    sequences the RNN updates its states and produces output sequences.
    Thus the input variables of your transition function play one of
    three roles: an input, a context or a state. These roles should be
    specified in the method's signature to make iteration possible.

    Parameters
    ----------
    inputs : list of strs
        Names of the arguments of the apply method that play input
        roles.
    states : list of strs
        Names of the arguments of the apply method that play state
        roles.
    contexts : list of strs
        Names of the arguments of the apply method that play context
        roles.
    outputs : list of strs
        Names of the outputs.

    """
    def recurrent_wrapper(application_function):
        arg_spec = inspect.getargspec(application_function)
        arg_names = arg_spec.args[1:]

        @wraps(application_function)
        def recurrent_apply(brick, application, application_call,
                            *args, **kwargs):
            """Iterates a transition function.

            Parameters
            ----------
            iterate : bool
                If ``True`` iteration is made. By default ``True``.
            reverse : bool
                If ``True``, the sequences are processed in backward
                direction. ``False`` by default.
            return_initial_states : bool
                If ``True``, initial states are included in the returned
                state tensors. ``False`` by default.

            .. todo::

                * Handle `updates` returned by the :func:`theano.scan`
                    routine.
                * ``kwargs`` has a random order; check if this is a
                    problem.

            """
            # Extract arguments related to iteration and immediately relay the
            # call to the wrapped function if `iterate=False`
            iterate = kwargs.pop('iterate', True)
            if not iterate:
                return application_function(brick, *args, **kwargs)
            reverse = kwargs.pop('reverse', False)
            return_initial_states = kwargs.pop('return_initial_states', False)

            # Push everything to kwargs
            for arg, arg_name in zip(args, arg_names):
                kwargs[arg_name] = arg

            # Make sure that all arguments for scan are tensor variables
            scan_arguments = (application.sequences + application.states +
                              application.contexts)
            for arg in scan_arguments:
                if arg in kwargs:
                    if kwargs[arg] is None:
                        del kwargs[arg]
                    else:
                        kwargs[arg] = tensor.as_tensor_variable(kwargs[arg])

            # Check which sequence and contexts were provided
            sequences_given = dict_subset(kwargs, application.sequences,
                                          must_have=False)
            contexts_given = dict_subset(kwargs, application.contexts,
                                         must_have=False)

            # Determine number of steps and batch size.
            if len(sequences_given):
                # TODO Assumes 1 time dim!
                shape = list(sequences_given.values())[0].shape
                if not iterate:
                    batch_size = shape[0]
                else:
                    n_steps = shape[0]
                    batch_size = shape[1]
            else:
                # TODO Raise error if n_steps and batch_size not found?
                n_steps = kwargs.pop('n_steps')
                batch_size = kwargs.pop('batch_size')

            # Handle the rest kwargs
            rest_kwargs = {key: value for key, value in kwargs.items()
                           if key not in scan_arguments}
            for value in rest_kwargs.values():
                if (isinstance(value, Variable) and not
                        is_shared_variable(value)):
                    logger.warning("unknown input {}".format(value) +
                                   unknown_scan_input)

            # Ensure that all initial states are available.
            for state_name in application.states:
                dim = brick.get_dim(state_name)
                if state_name in kwargs:
                    if isinstance(kwargs[state_name], NdarrayInitialization):
                        kwargs[state_name] = tensor.alloc(
                            kwargs[state_name].generate(brick.rng, (1, dim)),
                            batch_size, dim)
                    elif isinstance(kwargs[state_name], Application):
                        kwargs[state_name] = (
                            kwargs[state_name](state_name, batch_size,
                                               *args, **kwargs))
                else:
                    # TODO init_func returns 2D-tensor, fails for iterate=False
                    kwargs[state_name] = (
                        brick.initial_state(state_name, batch_size,
                                            *args, **kwargs))
                    assert kwargs[state_name]
            states_given = dict_subset(kwargs, application.states)

            # Theano issue 1772
            for name, state in states_given.items():
                states_given[name] = tensor.unbroadcast(state,
                                                        *range(state.ndim))

            def scan_function(*args):
                args = list(args)
                arg_names = (list(sequences_given) +
                             [output for output in application.outputs
                              if output in application.states] +
                             list(contexts_given))
                kwargs = dict(equizip(arg_names, args))
                kwargs.update(rest_kwargs)
                outputs = application(iterate=False, **kwargs)
                # We want to save the computation graph returned by the
                # `application_function` when it is called inside the
                # `theano.scan`.
                application_call.inner_inputs = args
                application_call.inner_outputs = pack(outputs)
                return outputs
            outputs_info = [
                states_given[name] if name in application.states
                else None
                for name in application.outputs]
            result, updates = theano.scan(
                scan_function, sequences=list(sequences_given.values()),
                outputs_info=outputs_info,
                non_sequences=list(contexts_given.values()),
                n_steps=n_steps,
                go_backwards=reverse)
            result = pack(result)
            if return_initial_states:
                # Undo Subtensor
                for i in range(len(states_given)):
                    assert isinstance(result[i].owner.op,
                                      tensor.subtensor.Subtensor)
                    result[i] = result[i].owner.inputs[0]
            if updates:
                application_call.updates = dict_union(application_call.updates,
                                                      updates)

            return result

        return recurrent_apply

    # Decorator can be used with or without arguments
    assert (args and not kwargs) or (not args and kwargs)
    if args:
        application_function, = args
        return application(recurrent_wrapper(application_function))
    else:
        def wrap_application(application_function):
            return application(**kwargs)(
                recurrent_wrapper(application_function))
        return wrap_application


class AERecurrent(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, **kwargs):
        super(AERecurrent, self).__init__(**kwargs)
        self.dim = dim
        self.children = [activation]

    @property
    def W(self):
        return self.params[0]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (AERecurrent.apply.sequences +
                    AERecurrent.apply.states):
            return self.dim
        return super(AERecurrent, self).get_dim(name)

    def _allocate(self):
        self.params.append(shared_floatx_nans((self.dim, self.dim), name="W"))

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states', 'curr_states', 'recons_states'], contexts=[])
    def apply(self, inputs=None, states=None, mask=None):
        transformed_states = tensor.dot(states, self.W)
        recons_states = tensor.dot(transformed_states,
                                   tensor.transpose(self.W))
        next_states = inputs + transformed_states
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states, states, recons_states
