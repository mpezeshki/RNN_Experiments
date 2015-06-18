from build_model import build_model
from train import train_model
from utils import parse_args

if __name__ == "__main__":
    args = parse_args()

    cost, cross_entropy = build_model(vocab_size, args)
    train_model(cost, cross_entropy, train_stream, valid_stream, args)
