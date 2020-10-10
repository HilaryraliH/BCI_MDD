import numpy as np
from model import EEGNet, ShallowConvNet, Pro_R1, Pro_R2, shallow_ShallowConvNet
from save_info import save_training_pic, check_path
from process_data_main import load_test_data, load_tr_val_data, get_save_path
import os
import csv

# set model config
model_name = "shallow_ShallowConvNet"  # EEGNet CRNN1  CRNN1_spatial
batch_size = 64
epoch = 50
verbose = 2
inter_sub_training = True

for val_sub in range(1, 2):

    ########################################################
    # get_save_path, save information path
    ########################################################
    save_dir = '.\\' + 'save_results' + '\\'
    check_path(save_dir)
    tr_data_file, tr_label_file, tr_label_binary_file, val_data_file, val_label_file, val_label_binary_file, test_data_file, test_file_list_file = get_save_path(
        val_sub)

    ########################################################
    # load training and validation data from file
    # Note: the data is randomlized, but later, we may don't need a randomlized dataset, this need to be modified
    ########################################################

    tr_data = np.load(tr_data_file)  # (1160,128,500,1)
    tr_label = np.load(tr_label_file)
    tr_label_binary = np.load(tr_label_binary_file)

    val_data = np.load(val_data_file)
    val_label = np.load(val_label_file)
    val_label_binary = np.load(val_label_binary_file)

    ########################################################
    # Plot the data
    ########################################################
    from matplotlib import pyplot as plt
    for i in range(128):
        sub1_sample = tr_data[20, i, :, 0]
        plt.plot(sub1_sample)

    plt.figure()
    for j in range(128):
        sub2_sample = tr_data[50, j, :, 0]
        plt.plot(sub2_sample)


    print(tr_label[10],tr_label[20],tr_label[30],tr_label[40],tr_label[50],tr_label[60])

    plt.show()

    ########################################################
    # train model, and evaluate model
    ########################################################
    model = eval(model_name)()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    if inter_sub_training:
        '''inter-sub training'''
        history = model.fit(tr_data, tr_label_binary, batch_size=batch_size, epochs=epoch, verbose=verbose,
                            shuffle=True, validation_data=(val_data, val_label_binary))
    else:
        '''intra-sub training'''
        history = model.fit(tr_data, tr_label_binary, batch_size=batch_size, epochs=epoch, verbose=verbose, \
                            shuffle=True, validation_split=0.1)

    '''保存模型 和 训练精度'''
    save_model_dir = save_dir + 'model\\'
    check_path(save_model_dir)
    model.save(save_model_dir + 'model' + str(val_sub) + '.h5')
    save_training_pic(history, save_dir, val_sub)

    ########################################################
    # Test the model and save the results to file
    ########################################################
    test_file_list = np.load(test_file_list_file)
    test_data = np.load(test_data_file)
    probs = model.predict(test_data)
    preds = probs.argmax(axis=-1)  # got the 0 or 1 label of test data

    # save results to file as the competition required
    f = open(save_dir + 'submit_result' + str(val_sub) + '.csv', 'w', newline='')
    result = csv.writer(f)
    results = [[i + 1, preds[i]] for i in range(len(preds))]
    result.writerows([['id', 'label']] + results)
    f.close()
