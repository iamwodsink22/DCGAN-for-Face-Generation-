import tensorflow as tf
import matplotlib.pyplot as plt

BATCH_SIZE = 256
train_images=tf.keras.preprocessing.image_dataset_from_directory('C:/Users/Dell/Downloads/archive (2)/img_align_celeba/img_align_celeba',labels=None,image_size=(28,28),batch_size=BATCH_SIZE)

normalization_layer = tf.keras.layers.Rescaling(scale=1./127.5,offset=-1)

train_images= train_images.map(lambda x: (normalization_layer(x)))

