from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Activation,
    Reshape, BatchNormalization, Concatenate)

def get_unet_batchnorm(input_shape, num_labels):
    num_rows, num_cols, num_layers = input_shape

    inputs = Input(input_shape)
    conv1 = BatchConvBlock(32, inputs, 1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BatchConvBlock(64, pool1, 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BatchConvBlock(128, pool2, 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = BatchConvBlock(256, pool3, 4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = BatchConvBlock(512, pool4, 5)

    up6 = Concatenate()([UpSampling2D(size=(2, 2))(conv5),conv4])
    conv6 = BatchConvBlock(256, up6, 6)

    up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6),conv3])
    conv7 = BatchConvBlock(128, up7, 7)

    up8 = Concatenate()([UpSampling2D(size=(2, 2))(conv7),conv2])
    conv8 = BatchConvBlock(64, up8, 8)

    up9 = Concatenate()([UpSampling2D(size=(2, 2))(conv8),conv1])
    conv9 = BatchConvBlock(32, up9, 9)

    conv10 = Conv2D(num_labels, (1,1), name='conv_10_1')(conv9)

    output_row = Reshape((num_rows*num_cols, num_labels))(conv10)
    output_row = Activation('softmax')(output_row)
    outputs = Reshape((num_rows, num_cols, num_labels))(output_row)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def BatchConvBlock(filters, inputs, block_num):
    """Create a conv block consisting of two conv layers with batch normalization

    """
    block = BatchConvStage(inputs, 1)
    block = BatchConvStage(block, 2)
    return block

def BatchConvStage(filters, inputs, block_num, stage_num):
    """Create conv layer with batch norm layer attached

    """
    conv_name = 'conv_{}_{}'.format(block_num, stage_num)
    stage = Conv2D(filters, (3,3), activation='relu', padding='same',
                    name=conv_name)(inputs)
    batch_name = 'batch_norm_{}_{}'.format(block_num, stage_num)
    stage = BatchNormalization(name=batch_name)(stage)
    stage = Activation('relu')(stage)

    return stage
