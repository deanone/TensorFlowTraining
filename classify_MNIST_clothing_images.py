import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys


def main():
	choice = int(sys.argv[1])

	# Load MNIST clothing images.
	fashion_mnist = keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

	# Examine data.
	print('Shapes of train images and labels:')
	print('Train images: ', train_images.shape)
	print('Train labels:', train_labels.shape)

	print('Shapes of test images and labels:')
	print('Test images:', test_images.shape)
	print('Test labels:', test_labels.shape)

	# Normalize data into the interval [0, 1]
	train_images = train_images / 255.0
	test_images = test_images / 255.0

	# Set up NN model.
	model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(256, activation = tf.nn.relu), keras.layers.Dense(10, activation = tf.nn.softmax)])

	# Compile model.
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	if choice == 1:
		# Train model.
		model.fit(train_images, train_labels, epochs = 5)

		# Evaluate model.
		test_loss, test_acc = model.evaluate(test_images, test_labels)

		print('Test accuracy: ', test_acc)
	else:
		epochs = [i for i in range(1, 10)]
		train_accs = []
		test_accs = []
		for epoch in epochs:
			# Train model.
			model.fit(train_images, train_labels, epochs = epoch)

			# Evaluate model.
			train_loss, train_acc = model.evaluate(train_images, train_labels)
			test_loss, test_acc = model.evaluate(test_images, test_labels)
			train_accs.append(train_acc)
			test_accs.append(test_acc)

		plt.plot(epochs, train_accs, 'xb-', label='training')
		plt.plot(epochs, test_accs, 'xr-', label='test')
		plt.xlabel('epochs')
		plt.ylabel('accuracy')
		plt.legend(loc='best')
		plt.show()


	return


if __name__ == '__main__':
	main()