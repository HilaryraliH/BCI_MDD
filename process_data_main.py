
########################################################
# leave one sub out
########################################################

import numpy as np
import pandas as pd
from keras.utils import to_categorical
import os
from save_info import check_path
import csv


########################################################
# load training set (40*sub_num samples, 128*500*1)
# Input: None
# Output: an array of (sampels, channels, sample_rates, 1)
# Other Info: N,normal:label=0   D,MDD:label=1
########################################################
def load_tr_val_data(sub_list, tr_data_dir):
    tr_data = []
    tr_label = []
    # sub_file_list = [os.path.join(tr_data_dir,str(i)) for i in sub_list]

    for sub_name in sub_list:
        print('     loading {}.csv'.format(sub_name), end='  ')
        tr_sub_path = os.path.join(tr_data_dir, sub_name)
        if sub_name[0] == 'D':
            sub_label = np.ones((len(os.listdir(tr_sub_path)),))  # 患者，label为1
        elif sub_name[0] == 'N':
            sub_label = np.zeros((len(os.listdir(tr_sub_path)),))  # 正常，label为0
        sub_data = np.zeros((len(os.listdir(tr_sub_path)), 128, 500))
        for i, file_name in enumerate(os.listdir(tr_sub_path)):
            file_path = os.path.join(tr_sub_path, file_name)
            file_data = np.array(pd.read_csv(file_path, header=None))
            sub_data[i] = file_data
        if tr_data == []:
            tr_data = sub_data
            tr_label = sub_label
        else:
            tr_data = np.concatenate((tr_data, sub_data), axis=0)
            tr_label = np.concatenate((tr_label, sub_label))
        print("     data.shape:{}, label.shape:{}".format(
            tr_data.shape, tr_label.shape))

    print("     Reshape the data and label......")
    tr_data = np.expand_dims(tr_data, axis=3)
    tr_label = tr_label.reshape((tr_label.shape[0], 1))
    tr_label_binary = to_categorical(tr_label)

    print("     Randomlize the index......   I can also do it in keras model.fit(...shuffle=True...), not here")
    rand_inx = np.random.permutation(range(tr_data.shape[0]))
    tr_data = tr_data[rand_inx]
    tr_label = tr_label[rand_inx]
    tr_label_binary = tr_label_binary[rand_inx]
    print("     successfully load tr_data {} tr_label {} tr_label_binary {}".format(
        tr_data.shape, tr_label.shape, tr_label_binary.shape))
    return tr_data, tr_label, tr_label_binary

########################################################
# load testidation set (440 samples, 128*500*1)
# Input: None
# Output: an array of (sampels, channels, sample_rates, 1)
########################################################


def load_test_data(test_data_dir):
    test_data = []
    test_file_list = os.listdir(test_data_dir)
    for file_name in test_file_list:
        file_path = os.path.join(test_data_dir, file_name)
        file_data = np.array(pd.read_csv(file_path, header=None))
        file_data = np.expand_dims(file_data, axis=0)
        if test_data == []:
            test_data = file_data
        else:
            test_data = np.concatenate((test_data, file_data), axis=0)
        if test_data.shape[0] % 50 == 0:
            print("     test_data.shape:{}".format(test_data.shape))
    test_data = np.expand_dims(test_data, axis=3)
    print("     Finnaly, test_data.shape:{}".format(test_data.shape))
    return test_file_list, test_data


def get_save_path(val_sub):
    # processed data directory
    processed_data_dir = '.\\'+'processed_data\\'
    check_path(processed_data_dir)

    tr_data_file = processed_data_dir+'tr_data' + str(val_sub) + '.npy'
    tr_label_file = processed_data_dir+'tr_label' + str(val_sub) + '.npy'
    tr_label_binary_file = processed_data_dir + \
        'tr_label_binary' + str(val_sub) + '.npy'

    val_data_file = processed_data_dir+'val_data' + str(val_sub) + '.npy'
    val_label_file = processed_data_dir+'val_label' + str(val_sub) + '.npy'
    val_label_binary_file = processed_data_dir + \
        'val_label_binary' + str(val_sub) + '.npy'

    test_data_file = processed_data_dir+'test_data' + str(val_sub) + '.npy'
    test_file_list_file = processed_data_dir + \
        'test_file_list' + str(val_sub) + '.npy'
    return tr_data_file, tr_label_file, tr_label_binary_file, val_data_file, val_label_file, val_label_binary_file, test_data_file, test_file_list_file

if __name__ == "__main__":

    total_D_num = 14  # 重度抑郁症人数（label为1）
    total_N_num = 17  # 正常被试人数（label为0）

    # source data path
    tr_root_dir = '.\\Training_Set'
    tr_data_dir = os.path.join(tr_root_dir, 'train_data')
    test_root_dir = '.\\Validation_Set'
    test_data_dir = os.path.join(test_root_dir, 'data')

    for val_sub in range(1, 15):
        
        # get the save path
        tr_data_file, tr_label_file, tr_label_binary_file, val_data_file, val_label_file, val_label_binary_file, test_data_file, test_file_list_file = get_save_path(
            val_sub)
        
        # get which sub for train, and which sub for vali
        val_sub_list = ['D'+str(val_sub), 'N'+str(val_sub)]
        tr_sub_list = ['D'+str(i) for i in range(1, val_sub)] + ['D'+str(j) for j in range(val_sub+1, total_D_num+1)] + [
            'N'+str(k) for k in range(1, val_sub)] + ['N'+str(l) for l in range(val_sub+1, total_N_num+1)]

        # load train/vali/test data
        print('Loading the \033[0;32;m{}th\033[0m as training data...'.format(
            tr_sub_list))
        tr_data, tr_label, tr_label_binary = load_tr_val_data(
            tr_sub_list, tr_data_dir)  # get train data
        print('Loading the \033[0;32;m{}th\033[0m as valuation data...'.format(
            val_sub_list))
        val_data, val_label, val_label_binary = load_tr_val_data(
            val_sub_list, tr_data_dir)  # get val data
        print('Loading the test data from \033[0;32;m{}th\033[0m (the test directiory)'.format(
            test_root_dir))
        test_file_list, test_data = load_test_data(test_data_dir)  # get test data

        # save to file
        np.save(tr_data_file, tr_data)
        np.save(tr_label_file, tr_label)
        np.save(tr_label_binary_file, tr_label_binary)

        np.save(val_data_file, val_data)
        np.save(val_label_file, val_label)
        np.save(val_label_binary_file, val_label_binary)

        np.save(test_file_list_file, test_file_list)# save the each name in a .csv, for save test result to submit
        np.save(test_data_file, test_data)
