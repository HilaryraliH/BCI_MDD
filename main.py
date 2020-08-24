import numpy as np
from model import EEGNet
from load_data import check_path
from save_info import save_training_pic

########################################################
# load training data
########################################################
root_dir = 'C:\\Users\\Impos\\Desktop\\MDD\\BCI_MDD\\'
processed_data_dir = root_dir+'processed_data\\'
tr_data=np.load(processed_data_dir+'tr_data.npy')
tr_label=np.load(processed_data_dir+'tr_label.npy')
tr_label_binary=np.load(processed_data_dir+'tr_label_binary.npy')


########################################################
# train
########################################################
save_dir = root_dir+'save_pic_info\\'
check_path(save_dir)

model = EEGNet()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(tr_data,tr_label_binary,batch_size=64,epochs=2,verbose=1,validation_split=0.1)
save_training_pic(history, save_dir)




'''train'''



'''evaluate'''
# val_file_list = np.load(processed_data_dir+'val_file_list.npy')
# val_data = np.load(processed_data_dir+'val_data.npy')

'''plot and save'''