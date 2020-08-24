'''输入为2D（chans，200，1）的模型'''
import keras.backend as K
from keras.layers import *
from keras.models import Model
from keras import optimizers
from keras.constraints import max_norm

model_input = Input(shape=(128,500,1))

def EEGNet():
    # article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
    # changed as the comments
    block1 = Conv2D(8, (1, 64), padding='same', use_bias=False)(model_input)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((128, 1), depth_multiplier=2, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(0.5)(block1)

    block2 = SeparableConv2D(16, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 4))(block2)  
    block2 = Dropout(0.5)(block2)

    flatten = Flatten()(block2)
    softmax = Dense(2, kernel_constraint=max_norm(0.25),activation='softmax')(flatten)

    return Model(inputs=model_input, outputs=softmax)