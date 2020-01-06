import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

def get_dataset(args):

	if args.dataset == 'mnist':
		mnist = tf.keras.datasets.mnist
		(x_train, y_train), _ = mnist.load_data()
		x_train = (x_train - 127.5) / 255.0
		x_train = x_train[..., tf.newaxis].astype(np.float32)
		args.data_shape = [28, 28, 1]

	train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(args.batch_size)

	return train_ds

def plot(images, save_dir, step):
    target_dir = os.path.join(save_dir, str(step))
    n = 8
    pad = 2
    width = images[0].shape[0]
    image = np.zeros(((width + pad) * n - pad, (width + pad) * n - pad))
    for i in range(n):
        for j in range(n):
            image[(width + pad) * i:(width + pad) * i + width, (width + pad) * j:(width + pad) * j + width] = images[i * n + j][:, :, 0]
    plt.imsave(os.path.join(save_dir, f'{step}.jpg'), image, cmap = 'gray')