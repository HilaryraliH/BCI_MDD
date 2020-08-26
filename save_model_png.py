import os
from keras.utils import plot_model
from config import *
from model import EEGNet,CRNN

os.environ["PATH"] += os.pathsep + 'C:/C1_Install_package/Graphviz/Graphviz 2.44.1/bin'
model = eval(model_name)()
plot_model(model,to_file=save_md_stru_dir+model_name+'.png',show_shapes=True)

# \033[0;32;m{}\033[0m