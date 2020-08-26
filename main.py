import numpy as np
from model import EEGNet, CRNN
from save_info import save_training_pic
from config import *

########################################################
# load training data
# Note: the data is randomlized, but later, we may don't need a randomlized dataset, this need to be modified
########################################################
tr_data=np.load(tr_data_file)
tr_label=np.load(tr_label_file)
tr_label_binary=np.load(tr_label_binary_file)

########################################################
# train
########################################################

model = eval(model_name)()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(tr_data,tr_label_binary,batch_size=batch_size,epochs=epochs,verbose=verbose,\
    shuffle=True, validation_split=0.1)

save_training_pic(history, save_dir)

########################################################
# evaluate
########################################################
val_file_list = np.load(val_file_list_file)
val_data = np.load(val_data_file)
val_pre = model.predict(val_data)
val_pre = val_pre[:,1] # got the 0 ro 1 label of validation data
# but I don't see the sample_submission.csv, don't know the format, so ..... wait...



########################################################
# plot and save
########################################################
