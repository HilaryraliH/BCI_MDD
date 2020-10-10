import matplotlib.pyplot as plt
import numpy as np
import os

def save_training_pic(hist, save_dir,val_sub):
    #creat acc pic
    save_acc_dir = save_dir+'acc\\'
    check_path(save_acc_dir)
    save_acc_file = save_acc_dir + str(val_sub)+'acc.png'

    plt.figure()
    metric = 'accuracy'
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric + ': {}'.format(hist.history['val_' + metric][-1]))
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(save_acc_file, bbox_inches='tight')
    plt.close()

    # create loss pic
    save_loss_dir = save_dir+'loss\\'
    check_path(save_loss_dir)
    save_loss_file = save_loss_dir + str(val_sub)+'loss.png'

    plt.figure()
    metric = 'loss'
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric + ': {}'.format(hist.history['val_' + metric][-1]))
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(save_loss_file, bbox_inches='tight')
    plt.close()


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
