import theano
import numpy
from theano import tensor
from blocks.model import Model
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
from blocks.bricks.recurrent import LSTM
from blocks.graph import ComputationGraph
from datasets import single_bouncing_ball, save_as_gif

floatX = theano.config.floatX


n_epochs = 90
x_dim = 225
h_dim = 600

print 'Building model ...'
# T x B x F
x = tensor.tensor3('x', dtype=floatX)
y = tensor.tensor3('y', dtype=floatX)

x_to_h = Linear(name='x_to_h',
                input_dim=x_dim,
                output_dim=4 * h_dim)
x_transform = x_to_h.apply(x)
lstm = LSTM(activation=Tanh(),
            dim=h_dim, name="lstm")
h, c = lstm.apply(x_transform)
h_to_o = Linear(name='h_to_o',
                input_dim=h_dim,
                output_dim=x_dim)
y_hat = h_to_o.apply(h)
sigm = Sigmoid()
y_hat = sigm.apply(y_hat)
y_hat.name = 'y_hat'

# only for generation
h_initial = tensor.tensor3('h_initial', dtype=floatX)
c_initial = tensor.tensor3('c_initial', dtype=floatX)
h_testing, c_testing = lstm.apply(x_transform, h_initial,
                                  c_initial, iterate=False)
y_hat_testing = h_to_o.apply(h_testing)
y_hat_testing = sigm.apply(y_hat_testing)
y_hat_testing.name = 'y_hat_testing'

cost = SquaredError().apply(y, y_hat)
cost.name = 'SquaredError'
# Initialization
for brick in (lstm, x_to_h, h_to_o):
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0)
    brick.initialize()

print 'Bulding training process...'
algorithm = GradientDescent(cost=cost,
                            params=ComputationGraph(cost).parameters,
                            step_rule=CompositeRule([StepClipping(10.0),
                                                     Scale(4)]))
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

generate1 = theano.function([x], [y_hat, h, c])
generate2 = theano.function([x, h_initial, c_initial],
                            [y_hat_testing, h_testing, c_testing])
initial_seq = inputs[0, :20, 0:1, :]
current_output, current_hidden, current_cell = generate1(initial_seq)
current_output = current_output[-1:]
current_hidden = current_hidden[-1:]
current_cell = current_cell[-1:]
generated_seq = initial_seq[:, 0]
next_input = current_output
prev_state = current_hidden
prev_cell = current_cell
for i in range(200):
    current_output, current_hidden, current_cell = generate2(next_input,
                                                             prev_state,
                                                             prev_cell)
    next_input = current_output
    prev_state = current_hidden
    prev_cell = current_cell
    generated_seq = numpy.vstack((generated_seq, current_output[:, 0]))
print generated_seq.shape
save_as_gif(generated_seq.reshape(generated_seq.shape[0],
                                  numpy.sqrt(generated_seq.shape[1]),
                                  numpy.sqrt(generated_seq.shape[1])))
