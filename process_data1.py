
########################################################
# Mix each sub
########################################################
import numpy as np
import pandas as pd 
from keras.utils import to_categorical
import os



########################################################
# load training set (1240 samples, 128*500*1)
# Input: None
# Output: an array of (sampels, channels, sample_rates, 1)
# Other Info: N,normal:label=0   D,MDD:label=1
########################################################

def load_tr_data():
    tr_data = []
    tr_label = []
    for cnt, sub_name in enumerate(os.listdir(tr_data_dir)):
        print('loading \033[0;32;m{}\033[0m.csv , \033[0;32;m{}\033[0m/\033[0;32;m{}\033[0m sub'.format(sub_name,cnt+1,len(os.listdir(tr_data_dir))),end='  ')
        tr_sub_path = os.path.join(tr_data_dir, sub_name)
        if sub_name[0]=='D':                                                    
            sub_label = np.ones((len(os.listdir(tr_sub_path)),))
        elif sub_name[0]=='N':
            sub_label = np.zeros((len(os.listdir(tr_sub_path)),))
        sub_data = np.zeros((len(os.listdir(tr_sub_path)),128,500))
        for i,file_name in enumerate(os.listdir(tr_sub_path)):
            file_path = os.path.join(tr_sub_path,file_name)
            file_data = np.array(pd.read_csv(file_path,header=None))
            sub_data[i] = file_data
        if tr_data==[]:
            tr_data = sub_data
            tr_label = sub_label
        else:
            tr_data = np.concatenate((tr_data,sub_data),axis=0)
            tr_label = np.concatenate((tr_label,sub_label))
        print("  data.shape:{}, label.shape:{}".format(tr_data.shape, tr_label.shape))
    print("Reshape the data and label......")
    tr_data = np.expand_dims(tr_data,axis=3)
    tr_label = tr_label.reshape((tr_label.shape[0], 1))
    tr_label_binary = to_categorical(tr_label)

    print("Randomlize the index......   I can also do it in keras model.fit(...shuffle=True...), not here")
    rand_inx = np.random.permutation(range(tr_data.shape[0]))
    tr_data = tr_data[rand_inx]
    tr_label = tr_label[rand_inx]
    tr_label_binary = tr_label_binary[rand_inx]
    print("successfully load tr_data \033[0;32;m{}\033[0m tr_label \033[0;32;m{}\033[0m tr_label_binary \033[0;32;m{}\033[0m".format(tr_data.shape, tr_label.shape, tr_label_binary.shape))
    return tr_data, tr_label, tr_label_binary

########################################################
# load testidation set (440 samples, 128*500*1)
# Input: None
# Output: an array of (sampels, channels, sample_rates, 1)
########################################################

def load_test_data():
    test_data = []
    test_file_list = os.listdir(test_data_dir)
    for file_name in test_file_list:
        file_path = os.path.join(test_data_dir,file_name)
        file_data = np.array(pd.read_csv(file_path,header=None))
        file_data = np.expand_dims(file_data,axis=0)
        if test_data==[]:
            test_data = file_data
        else:
            test_data = np.concatenate((test_data,file_data),axis=0)
        if test_data.shape[0]%50==0:
            print("test_data.shape:{}".format(test_data.shape))
    test_data = np.expand_dims(test_data,axis=3)
    print("Finnaly, test_data.shape:\033[0;32;m{}\033[0m".format(test_data.shape))
    return test_file_list,test_data


########################################################
# load and save data to file
########################################################
from config import *
if __name__ == "__main__":
    tr_data, tr_label, tr_label_binary = load_tr_data()
    test_file_list,test_data = load_test_data()
    np.save(tr_data_file,tr_data)
    np.save(tr_label_file,tr_label)
    np.save(tr_label_binary_file,tr_label_binary)
    np.save(test_file_list_file,test_file_list)
    np.save(test_data_file,test_data)

