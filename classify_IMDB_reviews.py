import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys


def main():
	choice = int(sys.argv[1])

	# Load data.
	imdb = keras.datasets.imdb
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

	# Explore data.
	print('Train data and labels shapes:')
	print('Train data: ', train_data.shape)
	print('Train labels: ', train_labels.shape)

	print('Test data and labels shapes:')
	print('Test data: ', test_data.shape)
	print('Test labels: ', test_labels.shape)

	# The data are preprocessed, but its feature vector has different length, i.e. the length of its vector is equal to the number
	# of words in the review. Also, each value in the feature vector corresponds to the index of the word in the dictionary (i.e. from 0 to 9999 in a dictionary of 10000 words)
	print('Length of first training feature vector: ', len(train_data[0]))
	print('Length of second training feature vector: ', len(train_data[1]))

	# Make feature vectors have equal length.
	word_index = imdb.get_word_index()
	word_index["<PAD>"] = 0
	word_index["<START>"] = 1
	word_index["<UNK>"] = 2  # unknown
	word_index["<UNUSED>"] = 3

	train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
	test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

	# Build model.
	vocab_size = 10000

	model = keras.Sequential()
	model.add(keras.layers.Embedding(vocab_size, 16))
	model.add(keras.layers.GlobalAveragePooling1D())
	model.add(keras.layers.Dense(16, activation=tf.nn.relu))
	model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

	# Compile model.
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

	# Create validation set from training set. The validation set contains the first 10000 samples.
	x_val = train_data[:10000]
	partial_x_train = train_data[10000:]

	y_val = train_labels[:10000]
	partial_y_train = train_labels[10000:]

	# Train model.
	history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

	# Evaluate model.
	results = model.evaluate(test_data, test_labels)
	print(results)

	# Create training history plots.
	history_dict = history.history
	acc = history_dict['acc']
	loss = history_dict['loss']
	val_acc = history_dict['val_acc']
	val_loss = history_dict['val_loss']

	epochs = range(1, len(acc) + 1)

	# Plot losses.
	plt.figure()

	if choice == 1:
		plt.plot(epochs, loss, 'bo', label='Training loss')
		plt.plot(epochs, val_loss, 'b', label='Validation loss')
		plt.title('Training and validation loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend(loc='best')
	else:
		plt.plot(epochs, acc, 'bo', label='Training accuracy')
		plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
		plt.title('Training and validation accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend(loc='best')
	plt.show()


	return


if __name__ == '__main__':
	main()