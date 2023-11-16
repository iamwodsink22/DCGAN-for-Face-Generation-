# DCGAN-for-Face-Generation-
Generate new faces using Deep Convolutional Generative Adversial Network.
[CelebA dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) from Kaggle was used for training.

It will take 200-250 epochs or even more for the generator to generate images resembling faces. Training can take a long time so for completing every 5 epochs a checkpoint is created so that training isn't lost.
Checkpoint upto 45 epochs is present in the repo so your first epoch will be 46th epoch 
To start training just run the file 'train.py' or execute command  'python -u train.py' on terminal.
Feel free to experiment on various input sizes, color channels, layers and units,learning rates as your liking.
