import keras.backend as K
from keras.layers import *
from keras.models import Model
from keras import optimizers
from keras.constraints import max_norm

model_input = Input(shape=(128,500,1))

def EEGNet():
    # article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
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


def CRNN1():
    permuted = Permute((3,2,1))(model_input) 
    block1 = Conv2D(64, (1, 20))(permuted)
    block1 = BatchNormalization(axis=-1)(block1)

    block1 = Conv2D(32, (1, 20))(block1)
    block1 = BatchNormalization()(block1) 
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(0.5)(block1)

    block2 = Conv2D(16, (1, 16) )(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)

    block3 = Reshape((int(block2.shape[-2]), int(block2.shape[-1])))(block2)

    lstm4 = LSTM(32, return_sequences=True)(block3)
    lstm4 = LSTM(8, return_sequences=True)(lstm4)

    flatten5 = Flatten()(lstm4)
    preds = Dense(2, activation='softmax', kernel_constraint=max_norm(0.25))(flatten5)

    return Model(inputs=model_input, outputs=preds)

def CRNN1_spatial():
    block1 = Conv2D(64, (128, 1))(model_input)
    block1 = Conv2D(64, (1, 20))(block1)
    block1 = BatchNormalization(axis=-1)(block1)

    block1 = Conv2D(32, (1, 20))(block1)
    block1 = BatchNormalization()(block1) 
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(0.5)(block1)

    block2 = Conv2D(16, (1, 16) )(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)

    block3 = Reshape((int(block2.shape[-2]), int(block2.shape[-1])))(block2)

    lstm4 = LSTM(32, return_sequences=True)(block3)
    lstm4 = LSTM(8, return_sequences=True)(lstm4)

    flatten5 = Flatten()(lstm4)
    preds = Dense(2, activation='softmax', kernel_constraint=max_norm(0.25))(flatten5)

    return Model(inputs=model_input, outputs=preds)


def Pro_R1():
    permuted = Permute((3, 2, 1))(model_input)
    block1 = Conv2D(8, (1, 20))(permuted)
    block1 = BatchNormalization(axis=-1)(block1)

    block1 = DepthwiseConv2D((1, 20), depth_multiplier=2, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1) 
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(0.5)(block1)

    block2 = SeparableConv2D(16, (1, 16) )(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)

    print(block2.shape)
    block3 = Reshape((int(block2.shape[-2]), int(block2.shape[-1])))(block2)

    lstm4 = LSTM(32, return_sequences=True)(block3)
    lstm4 = LSTM(8, return_sequences=True)(lstm4)

    flatten5 = Flatten()(lstm4)
    preds = Dense(2, activation='softmax', kernel_constraint=max_norm(0.25))(flatten5)

    return Model(inputs=model_input, outputs=preds)


def Pro_R2():
    permuted = Permute((3, 2, 1))(model_input)
    block1 = Conv2D(32, (1, 20))(permuted)
    block1 = BatchNormalization(axis=-1)(block1)

    block1 = DepthwiseConv2D((1, 20), depth_multiplier=2, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1) 
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(0.5)(block1)

    block2 = SeparableConv2D(16, (1, 16) )(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)

    print(block2.shape)
    block3 = Reshape((int(block2.shape[-2]), int(block2.shape[-1])))(block2)

    lstm4 = LSTM(32, return_sequences=True)(block3)
    lstm4 = LSTM(8, return_sequences=True)(lstm4)

    flatten5 = Flatten()(lstm4)
    preds = Dense(2, activation='softmax', kernel_constraint=max_norm(0.25))(flatten5)

    return Model(inputs=model_input, outputs=preds)


def ShallowConvNet():
    # article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
    # changed as the comments
    #block0 = BatchNormalization()(model_input)
    block1 = Conv2D(40, (1, 25), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(model_input)
    block1 = Conv2D(40, (128, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 75), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(0.5)(block1)
    flatten = Flatten()(block1)
    dense = Dense(2, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))
