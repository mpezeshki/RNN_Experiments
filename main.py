import theano
import numpy as np
from theano import tensor
from blocks.model import Model
from blocks.bricks import Linear, Tanh
from blocks.bricks.cost import SquaredError
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.bricks.recurrent import LSTM, SimpleRecurrent
from blocks.graph import ComputationGraph
from utils import learning_algorithm
from datasets import sum_of_sines2
from utils import plot_signals

floatX = theano.config.floatX


n_epochs = 40
x_dim = 1
h_dim = 3
o_dim = x_dim
num_batches = 30
batch_size = 20
num_time_steps = 150
depth = 3
teacher_force = False

print 'Building model ...'
# T x B x F
x = tensor.tensor3('x', dtype=floatX)
y = tensor.tensor3('y', dtype=floatX)

x_to_h1 = Linear(name='x_to_h1',
                 input_dim=x_dim,
                 output_dim=h_dim)
pre_rnn = x_to_h1.apply(x)
lstm = SimpleRecurrent(activation=Tanh(),
                       dim=h_dim, name="lstm")
h1 = lstm.apply(pre_rnn)
h1_to_o = Linear(name='h1_to_o',
                 input_dim=h_dim,
                 output_dim=o_dim)
y_hat = h1_to_o.apply(h1)
y_hat.name = 'y_hat'

# generation function
cg = ComputationGraph(y_hat)
generate = theano.function(inputs=cg.inputs,
                           outputs=y_hat)

cost = SquaredError().apply(y, y_hat)
cost.name = 'MSE'

# Initialization
for brick in (lstm, x_to_h1, h1_to_o):
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0)
    brick.initialize()

print 'Bulding training process...'
algorithm = GradientDescent(
    cost=cost,
    parameters=ComputationGraph(cost).parameters,
    step_rule=learning_algorithm(learning_rate=0.01, momentum=0.0,
                                 clipping_threshold=100, algorithm='adam'))

train_stream, valid_stream = sum_of_sines2(
    num_batches, batch_size, num_time_steps, depth=depth)

monitor_train_cost = TrainingDataMonitoring([cost],
                                            prefix="train",
                                            after_epoch=True)

monitor_valid_cost = DataStreamMonitoring([cost],
                                          data_stream=valid_stream,
                                          prefix="valid",
                                          after_epoch=True)

model = Model(cost)
main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                     extensions=[monitor_train_cost,
                                 monitor_valid_cost,
                                 FinishAfter(after_n_epochs=n_epochs),
                                 Printing()],
                     model=model)

print 'Starting training ...'
main_loop.run()

it = valid_stream.get_epoch_iterator()
for num in range(5):
    batch = it.next()
    sample_input_seq = batch[1][:, num:num + 1, :]
    sample_target_seq = batch[0][:, num:num + 1, :]

    if not teacher_force:
        sample_output_seq = generate(sample_input_seq)
    else:
        input_copy = sample_input_seq.copy()
        last = generate(input_copy[:30])[-1]
        input_copy[30] = last
        for i in np.arange(31, input_copy.shape[0]):
            last = generate(input_copy[:i])[-1]
            input_copy[i] = last
        sample_output_seq = generate(input_copy)

    plot_signals(sample_input_seq, sample_target_seq, sample_output_seq)
