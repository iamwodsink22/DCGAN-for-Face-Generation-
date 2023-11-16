import tensorflow as tf
from tensorflow import keras

import numpy



def generator_model():
    gen_model=keras.Sequential([
        keras.layers.Dense(7*7*256,use_bias=False,input_shape=(100,)),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Reshape((7,7,256)),
        keras.layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding='same',use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Conv2DTranspose(3,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh'),
       
        
        
        
        
    ])
    assert gen_model.output_shape==(None,28,28,3)
    return gen_model


def discrimator_model():
    disc_model=keras.Sequential([
        keras.layers.Conv2D(64,(5,5),strides=(2,2),padding='same',input_shape=[28,28,3]),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128,(5,5),strides=(2,2),padding='same'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(1),
        
        
    ])
    return disc_model