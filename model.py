import tensorflow as tf

class Discriminator(tf.keras.Model):
	def __init__(self, args, name = 'discriminator'):
		super(Discriminator, self).__init__(name = name)
		self.conv1 = tf.keras.layers.Conv2D(64, 7, 2, activation = tf.nn.leaky_relu)
		self.conv2 = tf.keras.layers.Conv2D(128, 7, 2, activation = tf.nn.leaky_relu)
		self.flatten = tf.keras.layers.Flatten()
		self.fc1 = tf.keras.layers.Dense(1)
		self.dis_layers = [self.conv1, self.conv2, self.flatten, self.fc1]
	def call(self, x):
		for layer in self.dis_layers:
			x = layer(x)
		return x

class Generator(tf.keras.Model):
	def __init__(self, args, name = 'generator'):
		super(Generator, self).__init__(name = name)
		self.fc1 = tf.keras.layers.Dense(7 * 7 * 128)
		self.reshape = tf.keras.layers.Reshape([7, 7, 128])
		self.deconv1 = tf.keras.layers.Conv2DTranspose(64, 7, 2, activation = tf.nn.leaky_relu, padding = 'same')
		self.deconv2 = tf.keras.layers.Conv2DTranspose(64, 7, 2, activation = tf.nn.leaky_relu, padding = 'same')
		self.deconv3 = tf.keras.layers.Conv2D(1, 3, 1, activation = tf.nn.tanh, padding = 'same')
		
		self.gen_layers = [self.fc1, self.reshape, self.deconv1, self.deconv2, self.deconv3]
	def call(self, x):
		for layer in self.gen_layers:
			x = layer(x)
		return x