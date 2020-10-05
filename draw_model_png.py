import os
from keras.utils import plot_model
from model import *
model_name="EEGNet" # EEGNet CRNN1  CRNN1_spatial
os.environ["PATH"] += os.pathsep + 'C:/C1_Install_package/Graphviz/Graphviz 2.44.1/bin'
model = eval(model_name)()
plot_model(model,to_file= model_name+'.png',show_shapes=True)

# \033[0;32;m{}\033[0m