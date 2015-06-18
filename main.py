import theano
import theano.tensor as tensor
from blocks.bricks import Linear
from blocks.bricks.lookup import LookupTable
from utils import parse_args, learning_algorithm
floatX = theano.config.floatX


def main(save_path, load_path, input_dim, state_dim, target_dim,
         epochs, learning_rate, momentum, clipping, algorithm,
         step_rule, **kwargs):
    print 'Building model ...'
    # Dimensions: <input_length> x <batch_size> x <num_features>
    x = tensor.tensor3('x', dtype=floatX)
    y = tensor.tensor3('y', dtype=floatX)

    x_to_h = Linear(name='x_to_h',
                    input_dim=x_dim,
                    output_dim=4 * h_dim)
    lookup = LookupTable(input_dim, dimension)
    x_transform = x_to_h.apply(x)
    lstm = LSTM(activation=Tanh(),
                dim=h_dim, name="lstm")
    h, c = lstm.apply(x_transform)
    h_to_o = Linear(name='h_to_o',
                    input_dim=h_dim,
                    output_dim=x_dim)
    y_hat = h_to_o.apply(h)
    y_hat = sigm.apply(y_hat)
    y_hat.name = 'y_hat'

    print 'Reading data ...'

if __name__ == '__main__':
    args = parse_args()
    step_rule = learning_algorithm(args)
    main(step_rule, **args.__dict__)
