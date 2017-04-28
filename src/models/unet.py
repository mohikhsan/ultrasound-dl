from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Activation,
    Reshape, Concatenate)

def get_unet(input_shape, num_labels):
    num_rows, num_cols, num_layers = input_shape

    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same', name='conv_1_1')(inputs)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same', name='conv_1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3,3), activation='relu', padding='same', name='conv_2_1')(pool1)
    conv2 = Conv2D(64, (3,3), activation='relu', padding='same', name='conv_2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3,3), activation='relu', padding='same', name='conv_3_1')(pool2)
    conv3 = Conv2D(128, (3,3), activation='relu', padding='same', name='conv_3_2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_4_1')(pool3)
    conv4 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_4_2')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3,3), activation='relu', padding='same', name='conv_5_1')(pool4)
    conv5 = Conv2D(512, (3,3), activation='relu', padding='same', name='conv_5_2')(conv5)

    up6 = Concatenate()([UpSampling2D(size=(2, 2))(conv5),conv4])
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_6_1')(up6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_6_2')(conv6)

    up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6),conv3])
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same', name='conv_7_1')(up7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same', name='conv_7_2')(conv7)

    up8 = Concatenate()([UpSampling2D(size=(2, 2))(conv7),conv2])
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same', name='conv_8_1')(up8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same', name='conv_8_2')(conv8)

    up9 = Concatenate()([UpSampling2D(size=(2, 2))(conv8),conv1])
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same', name='conv_9_1')(up9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same', name='conv_9_2')(conv9)

    conv10 = Conv2D(num_labels, (1,1), name='conv_10_1')(conv9)

    output_row = Reshape((num_rows*num_cols, num_labels))(conv10)
    output_row = Activation('softmax')(output_row)
    outputs = Reshape((num_rows, num_cols, num_labels))(output_row)

    model = Model(inputs=inputs, outputs=outputs)

    return model
