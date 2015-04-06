import theano
import numpy
from theano import tensor
from blocks.model import Model
from blocks.bricks import Linear, Tanh
from blocks.bricks.cost import SquaredError
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.graph import ComputationGraph
from datasets import single_bouncing_ball, save_as_gif

floatX = theano.config.floatX


n_epochs = 20
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
rnn = SimpleRecurrent(activation=Tanh(),
                      dim=h_dim, name="rnn")
h = rnn.apply(x_transform)
h_to_o = Linear(name='h_to_o',
                input_dim=h_dim,
                output_dim=x_dim)
y_hat = h_to_o.apply(h)
y_hat.name = 'y_hat'

cost = SquaredError().apply(y, y_hat)
cost.name = 'SquaredError'
# Initialization
for brick in (rnn, x_to_h, h_to_o):
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0)
    brick.initialize()

print 'Bulding training process...'
algorithm = GradientDescent(cost=cost,
                            params=ComputationGraph(cost).parameters,
                            step_rule=CompositeRule([StepClipping(10.0),
                                                     Scale(5)]))
monitor_cost = TrainingDataMonitoring([cost],
                                      prefix="train",
                                      after_epoch=True)

# S x T x B x F
inputs = single_bouncing_ball(10, 10, 200, 15, 2)
outputs = numpy.zeros(inputs.shape)
outputs[:, 0:-1, :, :] = inputs[:, 1:, :, :]
print 'Bulding DataStream ...'
dataset = IterableDataset({'x': inputs,
                           'y': outputs})
stream = DataStream(dataset)

model = Model(cost)
main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
                     extensions=[monitor_cost,
                                 FinishAfter(after_n_epochs=n_epochs),
                                 Printing()],
                     model=model)

print 'Starting training ...'
main_loop.run()

generate = theano.function([x], y_hat)
# takes 4 time steps
initial_seq = generate(inputs[0, :4, 0:1, :])
generated_seq = inputs[0, :4, 0, :]
# takes the last output after 4 time steps
next = initial_seq[-1:, :, :]
for i in range(200):
    next = generate(next)
    generated_seq = numpy.vstack((generated_seq, next[:, 0]))
print generated_seq.shape
save_as_gif(generated_seq.reshape(generated_seq.shape[0],
                                  numpy.sqrt(generated_seq.shape[1]),
                                  numpy.sqrt(generated_seq.shape[1])))
