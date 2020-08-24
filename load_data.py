import numpy as np
import pandas as pd 
import os
from keras.utils import to_categorical

########################################################
# check one path is exist, if not, make dir
########################################################
def check_path(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            print('make dir error')
            return

########################################################
# load training set (1240 samples, 128*500*1)
# Input: None
# Output: an array of (sampels, channels, sample_rates, 1)
# Other Info: N,normal:label=0   D,MDD:label=1
########################################################

def load_tr_data():
    tr_root_dir = '..\\Training_Set'
    tr_data_dir = os.path.join(tr_root_dir, 'train_data')
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
# load validation set (440 samples, 128*500)
# Input: None
# Output: an array of (sampels, channels, sample_rates, 1)
########################################################

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
    print("Finnaly, val_data.shape:\033[0;32;m{}\033[0m".format(tr_data.shape))
    return val_file_list,tr_data


########################################################
# load and save data to file
########################################################
if __name__ == "__main__":
    tr_data, tr_label, tr_label_binary = load_tr_data()
    val_file_list,val_data = load_val_data()

    root_dir = 'C:\\Users\\Impos\\Desktop\\MDD\\BCI_MDD\\'
    processed_data_dir = root_dir+'processed_data\\'
    check_path(processed_data_dir)

    np.save(processed_data_dir+'tr_data.npy',tr_data)
    np.save(processed_data_dir+'tr_label.npy',tr_label)
    np.save(processed_data_dir+'tr_label_binary.npy',tr_label_binary)
    np.save(processed_data_dir+'val_file_list.npy',val_file_list)
    np.save(processed_data_dir+'val_data.npy',val_data)

