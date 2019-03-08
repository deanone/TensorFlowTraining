import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def loss(y_predicted, y_real):
	return tf.reduce_mean(tf.square(y_predicted - y_real))


class Model(object):
	def __init__(self):
		self.W = tf.contrib.eager.Variable(-1.0)
		self.b = tf.contrib.eager.Variable(-1.0)

	def __call__(self, x):
		return self.W * x + self.b

	# Training the linear regression model using gradient descent.
	def train(self, inputs, outputs, learning_rate):
		with tf.GradientTape() as t:
			current_loss = loss(self(inputs), outputs)
		dW, db = t.gradient(current_loss, [self.W, self.b])
		self.W.assign_sub(learning_rate * dW)
		self.b.assign_sub(learning_rate * db)


def main():
	# Create random data
	TRUE_W = 3.0	# optimal value of W
	TRUE_b = 2.0	# optimal value of b
	N = 1000 # number of training samples

	inputs = tf.random_normal(shape=[N])	# Gaussian generated inputs
	noise = tf.random_normal(shape=[N])	# Gaussian generated nois
	outputs = inputs * TRUE_W + TRUE_b + noise	# outputs are linear regressions of inputs (with the optimal W and b) plus Gaussian noise

	model = Model()
	num_of_epochs = 50
	epochs = [epoch for epoch in range(num_of_epochs)]

	for epoch in epochs:
		model.train(inputs, outputs, learning_rate=0.1)
		plt.scatter(inputs, outputs, c='b', label='data')
		plt.scatter(inputs, model(inputs), c='r', label='model')
		plt.text(-3, 10, 'epoch {}'.format(epoch))
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title('Training of a simple linear regrassion model')
		plt.legend(loc='lower right')
		plt.draw()
		if epoch != (num_of_epochs - 1):
			plt.pause(0.1)	# time to pause (in seconds) between epoch plots
			plt.clf()
	#plt.show()

	return


if __name__ == '__main__':
	main()