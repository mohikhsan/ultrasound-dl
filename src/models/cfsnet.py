from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Activation,
    Reshape, Concatenate, Add)

def get_cfsnet(input_shape, num_labels):
    num_rows, num_cols, num_layers = input_shape

    inputs = Input(input_shape)
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same', name='conv_1_1')(inputs)
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same', name='conv_1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3,3), activation='relu', padding='same', name='conv_2_1')(pool1)
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same', name='conv_2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_3_1')(pool2)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_3_2')(conv3)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_3_3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_4_1')(pool3)
    conv4 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_4_2')(conv4)
    conv4 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_4_3')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_5_1')(pool4)
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_5_2')(conv5)
    conv5 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_5_3')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_6_1')(pool5)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_6_2')(conv6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same', name='conv_6_3')(conv6)

    c1 = UpSampling2D(size=(2, 2))(conv2)
    c1 = Conv2D(num_labels, (3,3), activation='relu', padding='same', name='c_1_1')(c1)
    c1 = Conv2D(num_labels, (1,1), activation='relu', padding='same', name='c_1_2')(c1)

    c2 = UpSampling2D(size=(4, 4))(conv3)
    c2 = Conv2D(num_labels, (3,3), activation='relu', padding='same', name='c_2_1')(c2)
    c2 = Conv2D(num_labels, (1,1), activation='relu', padding='same', name='c_2_2')(c2)

    c3 = UpSampling2D(size=(8, 8))(conv4)
    c3 = Conv2D(num_labels, (3,3), activation='relu', padding='same', name='c_3_1')(c3)
    c3 = Conv2D(num_labels, (1,1), activation='relu', padding='same', name='c_3_2')(c3)

    c4 = UpSampling2D(size=(16, 16))(conv5)
    c4 = Conv2D(num_labels, (3,3), activation='relu', padding='same', name='c_4_1')(c4)
    c4 = Conv2D(num_labels, (1,1), activation='relu', padding='same', name='c_4_2')(c4)

    c5 = UpSampling2D(size=(32, 32))(conv6)
    c5 = Conv2D(num_labels, (3,3), activation='relu', padding='same', name='c_5_1')(c5)
    c5 = Conv2D(num_labels, (1,1), activation='relu', padding='same', name='c_5_2')(c5)

    add1 = Add()([c1,c2,c3,c4,c5])
    output_row = Reshape((num_rows*num_cols, num_labels))(add1)
    output_row = Activation('softmax')(output_row)
    outputs = Reshape((num_rows, num_cols, num_labels))(output_row)

    model = Model(inputs=inputs, outputs=outputs)

    return model
