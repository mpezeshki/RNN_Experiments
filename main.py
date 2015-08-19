import theano
from theano import tensor
from blocks.model import Model
from blocks.bricks import Linear, Tanh, Logistic
from blocks.bricks.cost import SquaredError
from blocks.initialization import IsotropicGaussian, Constant
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from blocks.algorithms import GradientDescent
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.bricks.recurrent import LSTM
from blocks.graph import ComputationGraph
from utils import learning_algorithm
from datasets import random_signal_lag
from utils import plot_signals

floatX = theano.config.floatX


n_epochs = 200
x_dim = 1
h_dim = 10
o_dim = x_dim
num_batches = 10
batch_size = 15
num_time_steps = 20

print 'Building model ...'
# T x B x F
x = tensor.tensor3('x', dtype=floatX)
y = tensor.tensor3('y', dtype=floatX)

x_to_h1 = Linear(name='x_to_h1',
                 input_dim=x_dim,
                 output_dim=4 * h_dim)
pre_rnn = x_to_h1.apply(x)
lstm = LSTM(activation=Tanh(),
            dim=h_dim, name="lstm")
h1, c1 = lstm.apply(pre_rnn)
h1_to_o = Linear(name='h1_to_o',
                 input_dim=h_dim,
                 output_dim=o_dim)
pre_Logistic = h1_to_o.apply(h1)
# sigm = Logistic()
y_hat = pre_Logistic
# y_hat = sigm.apply(pre_Logistic)
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
    on_unused_sources='warn',
    cost=cost,
    parameters=ComputationGraph(cost).parameters,
    step_rule=learning_algorithm(learning_rate=0.01, momentum=0.0,
                                 clipping_threshold=100, algorithm='adam'))
monitor_cost = TrainingDataMonitoring([cost],
                                      prefix="train",
                                      after_epoch=True)

# S x T x B x F
input_seqs, target_seqs = random_signal_lag(
    num_batches, batch_size, num_time_steps)
print 'Bulding DataStream ...'
dataset = IterableDataset({'x': input_seqs,
                           'y': target_seqs})
stream = DataStream(dataset)

model = Model(cost)
main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
                     extensions=[monitor_cost,
                                 FinishAfter(after_n_epochs=n_epochs),
                                 Printing()],
                     model=model)

print 'Starting training ...'
main_loop.run()

batch = main_loop.data_stream.get_epoch_iterator().next()
sample_input_seq = batch[1]
sample_target_seq = batch[0]
sample_output_seq = generate(sample_input_seq)

plot_signals(sample_input_seq, sample_target_seq, sample_output_seq)
