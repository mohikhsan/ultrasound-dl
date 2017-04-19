"""
Define deep learning models
"""

from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Activation,
    Reshape, BatchNormalization, Concatenate)


def get_unet(input_shape, num_labels):
    print('Normal U-Net')

def get_unet_batchnorm(input_shape, num_labels):
    print('U-Net with Batch Normalization')

def get_conv_batchnorm_block():
    print('Create convolution block with batch normalization')
