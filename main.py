import numpy as np
from model import EEGNet
from save_info import save_training_pic,check_path
from process_data2 import load_test_data,load_tr_val_data
import os
import csv


# set model config
model_name="EEGNet" # EEGNet CRNN1  CRNN1_spatial
batch_size=128
epoch=2
verbose=2
total_D_num = 14 # 重度抑郁症人数（label为1）
total_N_num = 17 # 正常被试人数（label为0）

# project path
root_dir='C:\\Users\\Impos\\Desktop\\MDD\\BCI_MDD\\'
processed_data_dir='C:\\Users\\Impos\\Desktop\\MDD\\BCI_MDD\\'+'processed_data\\'
check_path(processed_data_dir)

# data path
tr_root_dir = '.\\Training_Set'
tr_data_dir = os.path.join(tr_root_dir, 'train_data')
test_root_dir = '.\\Validation_Set'
test_data_dir = os.path.join(test_root_dir, 'data')



########################################################
# Preprocess data
########################################################

for val_sub in range(1,15):

    ########################################################
    # get save_path
    ########################################################

    '''save information path'''
    save_dir = root_dir+'save_pic_info'+'\\'
    save_md_stru_dir = root_dir +'save_model_stru_png'+'\\'
    check_path(save_dir)
    check_path(save_md_stru_dir)

    '''save processed data path'''
    tr_data_file=processed_data_dir+'tr_data' + str(val_sub) +'.npy'
    tr_label_file=processed_data_dir+'tr_label' + str(val_sub) +'.npy'
    tr_label_binary_file=processed_data_dir+'tr_label_binary' + str(val_sub) +'.npy'

    val_data_file=processed_data_dir+'val_data' + str(val_sub) +'.npy'
    val_label_file=processed_data_dir+'val_label' + str(val_sub) +'.npy'
    val_label_binary_file=processed_data_dir+'val_label_binary' + str(val_sub) +'.npy'

    test_data_file=processed_data_dir+'test_data' + str(val_sub) +'.npy'
    test_file_list_file=processed_data_dir+'test_file_list' + str(val_sub) +'.npy'

    ########################################################
    # process data, and save to file
    ########################################################
    val_sub_list = ['D'+str(val_sub),'N'+str(val_sub)]
    tr_sub_list = ['D'+str(i) for i in range(1,val_sub)] + ['D'+str(j) for j in range(val_sub+1,total_D_num+1)] + ['N'+str(k) for k in range(1,val_sub)] + ['N'+str(l) for l in range(val_sub+1,total_N_num+1)]
    
    print('Loading the \033[0;32;m{}th\033[0m sub as training data...'.format(tr_sub_list))
    tr_data, tr_label, tr_label_binary = load_tr_val_data(tr_sub_list, tr_data_dir) # get train data
    print('Loading the \033[0;32;m{}th\033[0m sub as valuation data...'.format(val_sub_list))
    val_data, val_label, val_label_binary = load_tr_val_data(val_sub_list, tr_data_dir) # get val data
    print('Loading the test data from \033[0;32;m{}th\033[0m (the test directiory)'.format(test_root_dir))
    test_file_list,test_data = load_test_data(test_data_dir) # get test data

    np.save(tr_data_file,tr_data)
    np.save(tr_label_file,tr_label)
    np.save(tr_label_binary_file,tr_label_binary)

    np.save(val_data_file,val_data)
    np.save(val_label_file,val_label)
    np.save(val_label_binary_file,val_label_binary)

    np.save(test_file_list_file,test_file_list) # save the each name in a .csv, for save test result to submit
    np.save(test_data_file,test_data)

    ########################################################
    # load training data
    # Note: the data is randomlized, but later, we may don't need a randomlized dataset, this need to be modified
    ########################################################

    tr_data=np.load(tr_data_file)
    tr_label=np.load(tr_label_file)
    tr_label_binary=np.load(tr_label_binary_file)

    val_data=np.load(val_data_file)
    val_label=np.load(val_label_file)
    val_label_binary=np.load(val_label_binary_file)

    ########################################################
    # train model
    ########################################################
    model = eval(model_name)()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    '''intra-sub training'''
    # history = model.fit(tr_data,tr_label_binary,batch_size=batch_size,epochs=epoch,verbose=verbose,\
    #     shuffle=True, validation_split=0.1)

    '''inter-sub training'''
    history = model.fit(tr_data,tr_label_binary,batch_size=batch_size,epochs=epoch,verbose=verbose,\
        shuffle=True, validation_data=(val_data,val_label_binary))

    save_training_pic(history, save_dir,val_sub)

    ########################################################
    # Test the model and save the results to file
    ########################################################
    test_file_list = np.load(test_file_list_file)
    test_data = np.load(test_data_file)
    test_pre = model.predict(test_data)
    test_pre = test_pre[:,1] # got the 0 or 1 label of test data

    f = open('result1.csv','w',newline='') # save results to file as the competition required
    result = csv.writer(f)
    result.writerow(['id','label'])
    results = [[i,test_pre[i]] for i in range(len(test_pre))]
    f.close()
