import numpy as np
import pandas as pd 
import os
from keras.utils import to_categorical

########################################################
# load training set (1240 samples, 128*500*1)
# Input: None
# Output: an array of (sampels, channels, sample_rates, 1)
# Other Info: N,normalm:label=0   D,MDD:label=1
########################################################

def load_tr_data():
    tr_root_dir = '..\\Training_Set'
    tr_data_dir = os.path.join(tr_root_dir, 'train_data')
    tr_data = []
    tr_label = []
    for cnt, sub_name in enumerate(os.listdir(tr_data_dir)):
        print('loading {}.csv , {}/{} sub'.format(sub_name,cnt+1,len(os.listdir(tr_data_dir))),end='  ')
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
        print("data.shape:{}, label.shape:{}".format(tr_data.shape, tr_label.shape))
    print("Reshape the data and label......")
    tr_data = np.expand_dims(tr_data,axis=3)
    tr_label = tr_label.reshape((tr_label.shape[0], 1))
    train_y = to_categorical(tr_label)

    print("Randomlize the index......")
    rand_inx = np.random.permutation(range(tr_data.shape[0]))
    tr_data = tr_data[rand_inx]
    tr_label = tr_label[rand_inx]
    print("successfully load training data {} training label {}".format(tr_data.shape, tr_label.shape))
    return tr_data, tr_label

########################################################
# load validation set (440 samples, 128*500)
# Input: None
# Output: an array of (sampels, channels, sample_rates, 1)
########################################################'''erect model'''

def load_val_data():
    tr_root_dir = '..\\Validation_Set'
    tr_data_dir = os.path.join(tr_root_dir, 'data')
    tr_data = []
    val_file_list = os.listdir(tr_data_dir)
    for file_name in val_file_list:
        file_path = os.path.join(tr_data_dir,file_name)
        file_data = np.array(pd.read_csv(file_path,header=None))
        file_data = np.expand_dims(file_data,axis=0)
        if tr_data==[]:
            tr_data = file_data
        else:
            tr_data = np.concatenate((tr_data,file_data),axis=0)
        if tr_data.shape[0]%50==0:
            print("val_data.shape:{}".format(tr_data.shape))
    tr_data = np.expand_dims(tr_data,axis=3)
    print("Finnaly, val_data.shape:{}".format(tr_data.shape))
    return val_file_list,tr_data