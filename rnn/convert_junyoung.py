# WARNING: The code is ad-hoc for a special purpose and
#          as a result is so messy.
import numpy as np

jun_data = np.load('/data/lisa/data/wikipedia-text/enwiki_char_and_word.npz')
elo_data = np.load('/data/lisa/data/wikipedia-text/char_level_enwik8.npz')
new_data = {}

new_data['vocab'] = elo_data['vocab']
new_data['vocab_size'] = elo_data['vocab_size']
new_data['oov'] = elo_data['oov']

print jun_data['train_chars']

new_data['train'] = jun_data['train_chars']
new_data['valid'] = jun_data['valid_chars']
new_data['test'] = jun_data['test_chars']

mapping_jun_to_elo = np.array([90, 161, 172, 168, 130, 162, 166, 173, 0, 152,
                               146, 122, 51, 155, 26, 145, 35, 109, 63, 153,
                               62, 117, 42, 139, 192, 80, 56, 194, 60, 183,
                               66, 129, 54, 197, 40, 83, 102, 187, 92, 193,
                               94, 11, 164, 10, 179, 9, 196, 17, 2, 49, 30,
                               88, 136, 140, 147, 128, 114, 148, 135, 143,
                               82, 111, 23, 124, 107, 175, 32, 95, 58, 133,
                               67, 91, 28, 105, 101, 37, 134, 61, 204, 77,
                               189, 73, 78, 41, 199, 69, 110, 15, 202, 79,
                               89, 12, 181, 18, 177, 33, 163, 5, 200, 8, 24,
                               72, 106, 159, 154, 20, 144, 149, 126, 108, 156,
                               104, 84, 151, 29, 97, 47, 116, 25, 121, 44, 112,
                               65, 127, 38, 113, 43, 52, 131, 59, 165, 76, 201,
                               68, 86, 64, 190, 81, 125, 85, 150, 118, 22, 120,
                               19, 180, 74, 186, 6, 7, 27, 70, 176, 100, 160,
                               158, 169, 185, 178, 170, 174, 167, 96, 171, 93,
                               138, 50, 137, 39, 157, 31, 123, 57, 141, 46,
                               119, 34, 115, 36, 132, 55, 198, 48, 182, 53, 98,
                               103, 195, 75, 184, 87, 191, 14, 203, 3, 99, 1,
                               142, 4, 188, 16, 45, 13, 21, 71])
# import ipdb; ipdb.set_trace()
# np.savez('/data/lisatmp3/zablocki/jun_data', **new_data)


data_1 = np.load('/data/lisatmp3/zablocki/4XLSTM_700Units_ADAM/best')
data_3 = {}
tot_num_params = 0
for name in data_1.keys():
    if ('w' in name.lower()) or ('b' in name.lower()):
        data_3[name] = data_1[name]
        print name + "\t:\t" + str(data_1[name].shape)
        if len(data_1[name].shape) == 1:
            tot_num_params += data_1[name].shape[0]
        else:
            tot_num_params += data_1[name].shape[0] * data_1[name].shape[1]
print 'tot_num_params : ' + str(tot_num_params)

data_2 = np.load('/u/pezeshki/el/saved.npz')
tot_num_params_jun = 0
for i in range(len(data_2['arr_0'])):
    print str(data_2['arr_0'][i].shape)
    if len(data_2['arr_0'][i].shape) == 1:
        tot_num_params_jun += data_2['arr_0'][i].shape[0]
    else:
        tot_num_params_jun += (data_2['arr_0'][i].shape[0] *
                               data_2['arr_0'][i].shape[1])
print 'tot_num_params_jun : ' + str(tot_num_params_jun)

data_2 = data_2['arr_0']

data_3['output_layer.b'] = data_2[12]  # softmax_b
data_3['output_layer.W'] = data_2[13]  # softmax_W

data_3['recurrentstack-lstm_0.W_state'] = data_2[2]  # h1_W
data_3['recurrentstack-lstm_1.W_state'] = data_2[5]  # h2_W
data_3['recurrentstack-lstm_2.W_state'] = data_2[8]  # h3_W
data_3['recurrentstack-lstm_3.W_state'] = data_2[11]  # h4_W

data_3['fork-fork_inputs-lookuptable.b_lookup'] = data_2[1]  # p1_b
data_3['fork-fork_inputs_1-lookuptable.b_lookup'] = data_2[4]  # p2_b
data_3['fork-fork_inputs_2-lookuptable.b_lookup'] = data_2[7]  # p3_b
data_3['fork-fork_inputs_3-lookuptable.b_lookup'] = data_2[10]  # p4_b

data_3['fork-fork_inputs-lookuptable.W_lookup'] = data_2[0]  # p1_W
data_3['fork-fork_inputs_1-lookuptable.W_lookup'] = data_2[3][700:, :]  # p2_W
data_3['fork-fork_inputs_2-lookuptable.W_lookup'] = data_2[6][700:, :]  # p3_W
data_3['fork-fork_inputs_3-lookuptable.W_lookup'] = data_2[9][700:, :]  # p4_W

data_3['recurrentstack-fork_1-fork_inputs.W'] = data_2[3][:700, :]  # p4_W
data_3['recurrentstack-fork_2-fork_inputs.W'] = data_2[6][:700, :]  # p4_W
data_3['recurrentstack-fork_3-fork_inputs.W'] = data_2[9][:700, :]  # p4_W

np.savez('/data/lisatmp3/zablocki/4XLSTM_700Units_ADAM/best_jun', **data_3)

data_3 = np.load('/data/lisatmp3/zablocki/4XLSTM_700Units_ADAM/best_jun.npz')
print len(data_3.items())
