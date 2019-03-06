import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys


def norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']


def build_model(feature_vectors_len):
	model = keras.Sequential()
	model.add(keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[feature_vectors_len]))
	model.add(keras.layers.Dense(64, activation=tf.nn.relu))
	model.add(keras.layers.Dense(1))

	optimizer = tf.keras.optimizers.RMSprop(0.001)
	model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])		
	return model


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


def main():
	# Load data from UCI ML repo (https://archive.ics.uci.edu/ml/datasets/auto+mpg). 
	dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

	column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'Model Year', 'Origin']
	raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
	dataset = raw_dataset.copy()

	# Drop rows with unknown values
	dataset = dataset.dropna()

	# One-hot-encode 'Origin' feature
	origin = dataset.pop('Origin')
	dataset['USA'] = (origin == 1) * 1.0
	dataset['Europe'] = (origin == 2) * 1.0
	dataset['Japan'] = (origin == 3) * 1.0

	# Split dataset into train and test datasets.
	train_dataset = dataset.sample(frac=0.8,random_state=0)
	test_dataset = dataset.drop(train_dataset.index)

	# Explore data (data now are represented as Pandas dataframe).
	dataset.head()

	# Plot pairplot of variables in training dataset (optional)
	#sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

	# Print descriptive statistics of the data.
	train_stats = train_dataset.describe()
	train_stats.pop("MPG")
	train_stats = train_stats.transpose()
	print(train_stats)

	# Get targets to predict.
	train_labels = train_dataset.pop('MPG')
	test_labels = test_dataset.pop('MPG')

	# Standardize data.
	normed_train_data = norm(train_dataset, train_stats)
	normed_test_data = norm(test_dataset, train_stats)

	# Build model.
	model = build_model(len(train_dataset.keys()))

	# Train model.
	history = model.fit(normed_train_data, train_labels, epochs=100, validation_split=0.2, verbose=1)

	plot_history(history)

	# Evaluate on test data.
	loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
	print('Testing set Mean Abs Error: {:5.2f} MPG'.format(mae))

	# Make and plot predictions.
	test_predictions = model.predict(normed_test_data).flatten()

	plt.scatter(test_labels, test_predictions)
	plt.xlabel('True Values [MPG]')
	plt.ylabel('Predictions [MPG]')
	plt.axis('equal')
	plt.axis('square')
	plt.xlim([0,plt.xlim()[1]])
	plt.ylim([0,plt.ylim()[1]])
	_ = plt.plot([-100, 100], [-100, 100])
	plt.show()

	# Plot error distribution.
	error = test_predictions - test_labels
	plt.hist(error, bins = 25)
	plt.xlabel("Prediction Error [MPG]")
	_ = plt.ylabel("Count")
	plt.show()



	return


if __name__ == '__main__':
	main()