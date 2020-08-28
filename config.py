import os

def check_path(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            print('make dir error')
            return

# set model config
model_name = "Pro_R2" # EEGNet CRNN1  CRNN1_spatial
batch_size=64
epochs = 30
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
save_md_stru_dir = root_dir +'save_model_stru_png\\'
check_path(save_dir)
check_path(save_md_stru_dir)

# save processed data path
tr_data_file=processed_data_dir+'tr_data.npy'
tr_label_file=processed_data_dir+'tr_label.npy'
tr_label_binary_file=processed_data_dir+'tr_label_binary.npy'
val_file_list_file=processed_data_dir+'val_file_list.npy'
val_data_file=processed_data_dir+'val_data.npy'



