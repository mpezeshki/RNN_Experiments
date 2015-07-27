import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def save(destination, train, valid, test):
    np.savez(destination,
             train=train,
             valid=valid,
             test=test,
             feature_size=1)

# TIMIT file stored locally
# Can be downloaded form:
# https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav
sequence = wavfile.read('LDC93S1.wav')[1]

length = 300
batch = 10000
samples = []
samples.append(sequence[4500: 4500 + length].astype('float32'))
samples.append(sequence[8500: 8500 + length].astype('float32'))
samples.append(sequence[11500: 11500 + length].astype('float32'))
samples.append(sequence[14500: 14500 + length].astype('float32'))

for sample in samples:
    sample /= np.max(np.abs(sample))

selected = samples[1]
train = np.array([selected for i in range(batch)]).T[:, :, np.newaxis]
valid = train[:, :batch / 2, :]
test = train[:, :batch / 2, :]

# Save the data
save("/data/lisa/data/random_signal/data",
     train,
     valid,
     test)

plt.plot(range(length), train[:, i, 0])
axes = plt.gca()
axes.set_ylim([-1, 1])
plt.grid()
plt.show()
