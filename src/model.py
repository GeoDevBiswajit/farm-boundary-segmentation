import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, ReLU, BatchNormalization, Input, MaxPooling2D, Activation, Conv2DTranspose, \
    concatenate
from keras import Model
from keras.utils import plot_model
from config import *

def conv_block(inputs, filters, kernel_size=3, padding='same', strides=1, n_block=2, use_batchnorm=True, activation='relu'):
    if use_batchnorm:
        use_bias = False
    else:
        use_bias = True
    x = inputs
    for _ in range(n_block):
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)

    return x

# Encoder Block
def encoder_block(inputs, filters:list, pool_size=2):
    skips = []
    x = inputs
    for f in filters:
        x = conv_block(x, f)
        skips.append(x)
        x = MaxPooling2D(pool_size=(pool_size,pool_size))(x)
    return x, skips
# Decoder Block
def decoder_block(encoded_input, filters:list, skips:list, activation='relu', kernel_size=3, strides=2, padding='same'):
    x = encoded_input
    for f, skip in zip(filters[::-1], skips[::-1]):
        x = Conv2DTranspose(f, kernel_size=kernel_size, padding=padding, strides=strides)(x)
        x = concatenate([x, skip])
        x = conv_block(x, f)
    return x

def build_unet(image_size, channels, filters, bottelneck_filter = 1024, num_target_class=2):
    inputs = Input(shape=(image_size, image_size, channels))
    if num_target_class==2:
        out_activation = 'sigmoid'
        out_channel = 1
    elif num_target_class > 2:
        out_activation = 'softmax'
        out_channel = num_target_class
    else:
        raise ValueError ('Number of target class can not be less than 2')
    x, skips = encoder_block(inputs, filters=filters)
    bottelneck = conv_block(x, filters=bottelneck_filter)
    x = decoder_block(bottelneck, filters=filters, skips=skips)
    output = Conv2D(out_channel, kernel_size=(1,1), activation=out_activation)(x)
    model = Model(
        inputs=inputs,
        outputs=output,
        name="UNet"
    )
    return model

if __name__ == '__main__':
    model = build_unet(image_size=512, channels=4, filters=[64,128,256,512])
    plot_model(
        model, 
        to_file='model_architecture.png', 
        show_shapes=True, 
        show_layer_names=True
    )
    # print(model.summary())