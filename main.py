from build_model import build_model
from dataset import get_minibatch_char
from train import train_model
from utils import parse_args

if __name__ == "__main__":
    vocab_size = 205
    args = parse_args()

    train_stream, valid_stream = get_minibatch_char()
    cost, cross_entropy = build_model(vocab_size, args)
    train_model(cost, cross_entropy, train_stream, valid_stream, args)
