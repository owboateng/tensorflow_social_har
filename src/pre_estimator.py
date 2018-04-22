import tensorflow as tf
import argparse
import data_util

def main(argv):
  args = parser.parse_args(argv[1:])

  (train_x, train_y), (test_x, test_y) = data_util.load_data()
  feat_columns = []
  for key in train_x .keys():
    feat_columns.append(tf.feature_column.numeric_column(key=key))
  
  classifier = tf.estimator.DNNClassifier(
    feature_columns=feat_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 6 classes.
    n_classes=6
  )

  classifier.train(
    input_fn=lambda:data_util.train_input_fn(train_x, train_y, args.batch_size),
    steps=args.train_steps
  )

  eval_result = classifier.evaluate(
    input_fn=lambda:data_util.eval_input_fn(test_x, test_y,
                                  args.batch_size))

  print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=100, type=int, help='batch size')
  parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)