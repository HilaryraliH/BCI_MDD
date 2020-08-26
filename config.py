from load_data import check_path
import os

# set model config
model_name = "CRNN" # EEGNet CRNN
batch_size=64
epochs = 2
verbose = 2

# project path
root_dir = 'C:\\Users\\Impos\\Desktop\\MDD\\BCI_MDD\\'
processed_data_dir = root_dir+'processed_data\\'
check_path(processed_data_dir)

# data path
tr_root_dir = '..\\Training_Set'
tr_data_dir = os.path.join(tr_root_dir, 'train_data')
val_root_dir = '..\\Validation_Set'
val_data_dir = os.path.join(val_root_dir, 'data')

# save information path
save_dir = root_dir+'save_pic_info\\'
check_path(save_dir)

# save processed data path
tr_data_file=processed_data_dir+'tr_data.npy'
tr_label_file=processed_data_dir+'tr_label.npy'
tr_label_binary_file=processed_data_dir+'tr_label_binary.npy'
val_file_list_file=processed_data_dir+'val_file_list.npy'
val_data_file=processed_data_dir+'val_data.npy'



