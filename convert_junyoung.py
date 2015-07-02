import numpy as np

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
