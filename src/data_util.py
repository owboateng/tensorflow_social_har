import pandas as pd
import tensorflow as tf
import os

CSV_COLUMN_NAMES = map(str, range(1, 562))

def load_data():
  base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
  train_x_file = os.path.join(base_dir, 'data/final_X_train.txt')
  train_y_file = os.path.join(base_dir, 'data/final_y_train.txt')
  train_x = pd.read_csv(train_x_file, names=CSV_COLUMN_NAMES)
  train_y = pd.read_csv(train_y_file, names=['Activity']).pop('Activity')

  test_x_file = os.path.join(base_dir, 'data/final_X_test.txt')
  test_y_file = os.path.join(base_dir, 'data/final_y_test.txt') 
  test_x = pd.read_csv(test_x_file, names=CSV_COLUMN_NAMES)
  test_y = pd.read_csv(test_y_file, names=['Activity']).pop('Activity')

  return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
  """An input function for training"""
  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
  # Shuffle, repeat, and batch the examples.
  dataset = dataset.shuffle(1000).repeat().batch(batch_size)
  # Return the dataset.
  return dataset

def eval_input_fn(features, labels, batch_size):
  """An input function for evaluation or prediction"""
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
  # Batch the examples
  assert batch_size is not None, "batch_size must not be None"
  dataset = dataset.batch(batch_size)

  # Return the dataset.
  return dataset