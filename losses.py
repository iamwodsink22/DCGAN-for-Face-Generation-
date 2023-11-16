import tensorflow as tf
entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_img):
    return entropy(tf.ones_like(fake_img),fake_img)

def discrimantor_loss(real_img,fake_img):
    real_loss=entropy(tf.ones_like(real_img),real_img)
    fake_loss=entropy(tf.zeros_like(fake_img),fake_img)
    return real_loss+fake_loss
    
    