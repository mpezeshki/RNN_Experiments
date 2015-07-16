from rnn.build_model.build_model_vanilla import build_model_vanilla
from rnn.build_model.build_model_lstm import build_model_lstm
from rnn.build_model.build_model_cw import build_model_cw
from rnn.build_model.build_model_soft import build_model_soft
from rnn.build_model.build_model_hard import build_model_hard
from rnn.datasets.dataset import get_minibatch_char
from rnn.train import train_model
from rnn.utils import parse_args
from rnn.visualize import run_visualizations

if __name__ == "__main__":

    args = parse_args()

    # Choose the correct plotting backend
    import matplotlib
    if not args.local:
        matplotlib.use('Agg')

    dataset = args.dataset
    mini_batch_size = args.mini_batch_size
    mini_batch_size_valid = args.mini_batch_size_valid
    time_length = args.time_length
    rnn_type = args.rnn_type

    # Make sure we don't have skip_connections with only one hidden layer
    assert(not(args.skip_connections and args.layers == 1))

    # Prepare data
    train_stream, valid_stream, vocab_size = get_minibatch_char(
        dataset, mini_batch_size, mini_batch_size_valid,
        time_length, args.tot_num_char)

    # Build the model
    gate_values = None
    if rnn_type == "simple":
        cost, cross_entropy, updates, hidden_states = build_model_vanilla(
            vocab_size, args)
    elif rnn_type == "clockwork":
        cost, cross_entropy, updates, hidden_states = build_model_cw(
            vocab_size, args)
    elif rnn_type == "lstm":
        (cost, cross_entropy, updates, gate_values, hidden_states) = build_model_lstm(
            vocab_size, args)
    elif rnn_type == "soft":
        (cost, cross_entropy, updates, gate_values, hidden_states) = build_model_soft(
            vocab_size, args)
    elif rnn_type == "hard":
        cost, cross_entropy, updates, hidden_states = build_model_hard(
            vocab_size, args)
    else:
        assert(False)

    # Train the model
    if args.visualize == "nothing":
        train_model(cost, cross_entropy, updates,
                    train_stream, valid_stream,
                    args,
                    gate_values=gate_values)
    else:
        run_visualizations(cost, updates,
                           train_stream, valid_stream,
                           args,
                           hidden_states=hidden_states,
                           gate_values=gate_values,)
