import theano
import numpy
from theano import tensor
from blocks.bricks import Linear, Tanh, Sigmoid
from blocks.bricks.cost import SquaredError
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
# from blocks.bricks.recurrent import SimpleRecurrent
from models import AERecurrent
from blocks.graph import ComputationGraph
from datasets import single_bouncing_ball, save_as_gif
# import pylab
# import matplotlib.cm as cm
# m = pylab.figure()
# m.show()


floatX = theano.config.floatX


# @theano.compile.ops.as_op(itypes=[tensor.dmatrix],
#                           otypes=[tensor.dmatrix])
# def test_func(x):
#     s = numpy.dot(x.T, x)
#     s = 1 / (1 + numpy.exp(-5 * x))
#     pylab.imshow(s, interpolation='nearest', cmap=cm.Greys_r)
#     pylab.draw()
#
#     return x

n_epochs = 150
x_dim = 225
h_dim = 600

print 'Building model ...'
# T x B x F
x = tensor.tensor3('x', dtype=floatX)
y = tensor.tensor3('y', dtype=floatX)

x_to_h = Linear(name='x_to_h',
                input_dim=x_dim,
                output_dim=h_dim)
x_transform = x_to_h.apply(x)
ae_rnn = AERecurrent(activation=Tanh(),
                     dim=h_dim, name='ae_rnn')
h, h_prev, recons_h = ae_rnn.apply(x_transform)
h_to_o = Linear(name='h_to_o',
                input_dim=h_dim,
                output_dim=x_dim)
y_hat = h_to_o.apply(h)
sigm = Sigmoid()
y_hat = sigm.apply(y_hat)
y_hat.name = 'y_hat'

# only for generation B x h_dim
h_initial = tensor.tensor3('h_initial', dtype=floatX)
h_testing, _, _ = ae_rnn.apply(x_transform, h_initial, iterate=False)
y_hat_testing = h_to_o.apply(h_testing)
y_hat_testing = sigm.apply(y_hat_testing)
y_hat_testing.name = 'y_hat_testing'


_lambda = 0.0
generation_cost = SquaredError().apply(y, y_hat)
generation_cost.name = 'generation_cost'
ae_cost = SquaredError().apply(h_prev, recons_h)
ae_cost.name = 'ae_cost'
cost = generation_cost + _lambda * ae_cost
cost.name = 'SquaredError'
# Initialization
for brick in (x_to_h, h_to_o):
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0)
    brick.initialize()
ae_rnn.weights_init = IsotropicGaussian(0.01)
ae_rnn.biases_init = Constant(0)
ae_rnn.initialize()

print 'Bulding training process...'
algorithm = GradientDescent(cost=cost,
                            params=ComputationGraph(cost).parameters,
                            step_rule=CompositeRule([StepClipping(10.0),
                                                     Scale(0.1)]))
monitor_cost = TrainingDataMonitoring([cost],
                                      prefix="train",
                                      after_epoch=True)
monitor_g_cost = TrainingDataMonitoring([generation_cost],
                                        prefix="train_g",
                                        after_epoch=True)
monitor_r_cost = TrainingDataMonitoring([ae_cost],
                                        prefix="train_r",
                                        after_epoch=True)
# test = test_func(ae_rnn.params[0])
# test.name = 'test'
# monitor__test = TrainingDataMonitoring([test],
#                                        prefix="test",
#                                        after_epoch=True)

# S x T x B x F
inputs = single_bouncing_ball(10, 10, 200, 15, 2)
outputs = numpy.zeros(inputs.shape)
outputs[:, 0:-1, :, :] = inputs[:, 1:, :, :]
print 'Bulding DataStream ...'
dataset = IterableDataset({'x': inputs,
                           'y': outputs})
stream = DataStream(dataset)

main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
                     extensions=[monitor_cost,
                                 monitor_g_cost,
                                 monitor_r_cost,
                                 FinishAfter(after_n_epochs=n_epochs),
                                 Printing()])

print 'Starting training ...'
main_loop.run()

generate1 = theano.function([x], [y_hat, h])
generate2 = theano.function([x, h_initial], [y_hat_testing, h_testing])
initial_seq = inputs[0, :20, 0:1, :]
current_output, current_hidden = generate1(initial_seq)
current_output, current_hidden = current_output[-1:], current_hidden[-1:]
generated_seq = initial_seq[:, 0]
next_input = current_output
prev_state = current_hidden
for i in range(200):
    current_output, current_hidden = generate2(next_input, prev_state)
    next_input = current_output
    prev_state = current_hidden
    generated_seq = numpy.vstack((generated_seq, current_output[:, 0]))
print generated_seq.shape
save_as_gif(generated_seq.reshape(generated_seq.shape[0],
                                  numpy.sqrt(generated_seq.shape[1]),
                                  numpy.sqrt(generated_seq.shape[1])))
