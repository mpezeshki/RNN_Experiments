from datasets import random_signal_lag, sum_of_sines
from utils import plot_signals


# Test signal dataset and plotting
# input_seqs, target_seqs = random_signal_lag(1, 1, 25)
# output_seqs = target_seqs
# plot_signals(input_seqs[0], target_seqs[0], output_seqs[0])

sum_of_sines(10, 64, 50)
