from build_model import build_model
from dataset import get_minibatch_char
from train import train_model
from parser import parse_args

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset

    if dataset == "wikipedia":
        vocab_size = 205
    elif dataset == "penntree":
        vocab_size = 50

    mini_batch_size = args.mini_batch_size
    time_length = args.time_length

    train_stream, valid_stream = get_minibatch_char(dataset, mini_batch_size,
                                                    time_length)
    cost, cross_entropy = build_model(vocab_size, args)
    train_model(cost, cross_entropy, train_stream, valid_stream, args)
