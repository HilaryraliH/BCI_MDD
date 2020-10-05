import matplotlib.pyplot as plt
import numpy as np
import os

def save_training_pic(hist, save_dir):
    #creat acc directory
    save_acc_file = save_dir + 'acc.png'

    plt.figure()
    metric = 'accuracy'
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric + ': {}'.format(hist.history[metric][-1]))
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(save_acc_file, bbox_inches='tight')
    plt.close()

    # create loss directory
    save_loss_file = save_dir + 'loss.png'

    plt.figure()
    metric = 'loss'
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric + ': {}'.format(hist.history[metric][-1]))
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(save_loss_file, bbox_inches='tight')
    plt.close()


def check_path(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            print('make dir error')
            return
