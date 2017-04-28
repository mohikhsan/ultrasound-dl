from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Activation,
    Reshape, Concatenate)

def get_conv_logreg(input_shape, num_labels):
    num_rows, num_cols, num_layers = input_shape

    inputs = Input(input_shape)

    conv_out = Conv2D(num_labels, (3,3), padding='same',
        input_shape=input_shape, name='conv_out')

    output_row = Reshape((num_rows*num_cols, num_labels))(conv_out)
    output_row = Activation('softmax')(output_row)
    outputs = Reshape((num_rows, num_cols, num_labels))(output_row)

    model = Model(inputs=inputs, outputs=outputs)

    return model   
