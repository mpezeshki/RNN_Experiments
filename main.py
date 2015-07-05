from build_model_vanilla import build_model_vanilla
from build_model_lstm import build_model_lstm
from build_model_cw import build_model_cw
from build_model_soft import build_model_soft
from build_model_hard import build_model_hard
from dataset import get_minibatch_char
from train import train_model
from utils import parse_args

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    mini_batch_size = args.mini_batch_size
    mini_batch_size_valid = args.mini_batch_size_valid
    time_length = args.time_length
    rnn_type = args.rnn_type

    # Prepare data
    train_stream, valid_stream, vocab_size = get_minibatch_char(
        dataset, mini_batch_size, mini_batch_size_valid,
        time_length, args.tot_num_char)

    # Make sure we don't have skip_connections with only one hidden layer
    assert(not(args.skip_connections and args.layers == 1))

    # Build the model
    if rnn_type == "simple":
        cost, cross_entropy, updates = build_model_vanilla(vocab_size, args)
    elif rnn_type == "clockwork":
        cost, cross_entropy, updates = build_model_cw(vocab_size, args)
    elif rnn_type == "lstm":
        cost, cross_entropy, updates = build_model_lstm(vocab_size, args)
    elif rnn_type == "soft":
        cost, cross_entropy, updates = build_model_soft(vocab_size, args)
    elif rnn_type == "hard":
        cost, cross_entropy, updates = build_model_hard(vocab_size, args)
    else:
        assert(False)

    # Train the model
    train_model(cost, cross_entropy, updates, train_stream, valid_stream, args)
