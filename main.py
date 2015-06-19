from build_model import build_model
from dataset import get_minibatch_char
from train import train_model
from utils import parse_args

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset

    mini_batch_size = args.mini_batch_size
    time_length = args.time_length

    # Prepare data
    (train_stream,
     valid_stream,
     vocab_size) = get_minibatch_char(dataset, mini_batch_size, time_length)

    # Build the model
    cost, cross_entropy = build_model(vocab_size, args)

    # Train the model
    train_model(cost, cross_entropy, train_stream, valid_stream, args)
