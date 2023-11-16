import tensorflow as tf
from IPython import display
import os
import time
from mpl_toolkits.axes_grid1 import ImageGrid

from dataset import train_images
from models import generator_model,discrimator_model
from losses import generator_loss,discrimantor_loss
import matplotlib.pyplot as plt
EPOCHS = 50
noise_dim = 100
num_examples = 16
seed = tf.random.normal([num_examples, noise_dim])
discriminator=discrimator_model()
generator=generator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
BATCH_SIZE = 256
dir = './checkpoints'
checkpoint_prefix = os.path.join(dir, "checkpoint")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
@tf.function
def train_step(pics):
    noise=tf.random.normal([BATCH_SIZE,noise_dim])
    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
        generated_pics=generator(noise,training=True)
       
        fake_output=discriminator(generated_pics,training=True)
       
        real_output=discriminator(pics,training=True)
        
        gen_loss=generator_loss(fake_output)
        disc_loss=discrimantor_loss(real_output,fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
def save_img(model,epoch,input):
    predictions = model(input, training=False)

    

    predictions =predictions*127.5 +127.5
    predictions.numpy()
    images=[]
    os.mkdir(f"E:/GAN/epoch{epoch}")
    for i in range(num_examples):
        img = tf.keras.utils.array_to_img(predictions[i])
        images.append(img)
        img.save(f"epoch{epoch}/generated_img{i}.png")
        
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    for ax, im in zip(grid, images):
    # Iterating over the grid returns the Axes.
        ax.imshow(im)
        
    plt.savefig(f"images_at_epoch{epoch}.png")
    plt.show()
    
    
    
   



def train(dataset,epochs):
    checkpoint.restore(tf.train.latest_checkpoint(dir))
    for epoch in range(epochs):
        start = time.time()
        print(f"epoch {epoch+1}started in {start}")
        for image in dataset:
            train_step(image)
        display.clear_output(wait=True)
        save_img(generator,epoch+1,seed)
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

 
    display.clear_output(wait=True)
    save_img(generator,
                           epochs,
                           seed)


train(train_images, EPOCHS)
