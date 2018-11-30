import tensorflow as tf
import numpy as np
import os

#import vae
import na_vib
import error_utils
import validator

import glob
import joblib

import argparse


"""parsing and configuration"""
def parse_args():
  desc = "Tensorflow implementation of 'Negative Anchor Variational Information Bottleneck (NAVIB)'"
  parser = argparse.ArgumentParser(description=desc)

  #
  # negative_anchors
  #

  parser.add_argument('--lambda_param', type=float,default=0.001,
    help='negative anchor trade-off parameter')

  parser.add_argument('--beta_param', type=float,default=0.01,
    help='information bottleneck trade-off parameter')

  parser.add_argument('--keep_prob', type=float,default=0.9,
    help='training keep_prob for dropout')

  parser.add_argument('--save_freq', type=int,default=5,
    help='how often should we save parameters to file? (in epochs)')

  parser.add_argument('--param_save_path', default="params/",
    help='save path for params')

  parser.add_argument('--experiment_name', default=None,
    help='save prefix for params')

  parser.add_argument('--restart_epoch', type=int, default=-1,
    help='epoch to start at, uses param_save_path/experiment_name_#.ckpt')

  parser.add_argument('--data_path',default=None,\
    help='location of data (split dependency, sorry =.= )')

  parser.add_argument('--exp_override', default=False, action="store_true",
    help='overrides safety switch and writes over experiment files')

  parser.add_argument('--y_type', default="zero_one",
    help='defines error metric')

  parser.add_argument('--x_type', default="zero_one",
    help='defines error metric')

  #
  # vae
  #

  parser.add_argument('--dim_z', type=int, default='20', help='Dimension of latent vector', required = True)

  parser.add_argument('--n_hidden_xz',\
    type=int, default=64, help='Number of hidden units in MLP')

  parser.add_argument('--n_hidden_zy',\
    type=int, default=64, help='Number of hidden units in MLP')

  parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

  parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')

  parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

  return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

  # --experiment_name
  if not args.experiment_name:
    print("Experiment Needs a Name! Aborting.")
    return None

  # --param_save_path
  try:
      os.mkdir(args.param_save_path)
  except(FileExistsError):
      pass
  # delete all existing files
  files = glob.glob(args.param_save_path+'/*')
  if len(files) > 0 and not args.exp_override:
    print("Experiment params already exist! Either restart or use --exp_override flag.")
    return None
  elif args.exp_override:
    for f in files:
       os.remove(f)

  # --data_path
  if args.data_path is None:
    print("--data_path required.")
    return None

  # --dim-z
  try:
      assert args.dim_z > 0
  except:
      print('dim_z must be positive integer')
      return None

  # --n_hidden_xz
  try:
      assert args.n_hidden_xz >= 1
  except:
      print('number of hidden units must be larger than one')

  # --n_hidden_zy
  try:
      assert args.n_hidden_zy >= 1
  except:
      print('number of hidden units must be larger than one')

  # --learn_rate
  try:
      assert args.learn_rate > 0
  except:
      print('learning rate must be positive')

  # --num_epochs
  try:
      assert args.num_epochs >= 1
  except:
      print('number of epochs must be larger than or equal to one')

  # --batch_size
  try:
      assert args.batch_size >= 1
  except:
      print('batch size must be larger than or equal to one')

  return args

"""main function"""

def main(args):

  n_hidden_xz = args.n_hidden_xz
  n_hidden_zy = args.n_hidden_zy

  train_total_data, split_numbers, train_size,\
    validation_data, validation_labels, validation_confounds,\
    test_data, test_labels, test_confounds =\
    joblib.load( args.data_path )

  #maxes = np.maximum(np.maximum(\
  #  train_total_data[:,:split_numbers[0]].max(axis=0),\
  #  validation_data.max(axis=0)),\
  #  test_data.max(axis=0))

  n_samples = train_size 
  n_labels = split_numbers[1] - split_numbers[0]
  n_confounds = np.shape(train_total_data)[1] - split_numbers[1]

  """ parameters """

  dim_input = split_numbers[0]
  dim_out = n_labels
  batch_size = args.batch_size
  learn_rate = args.learn_rate
  num_epochs = args.num_epochs
  dim_z = args.dim_z

  beta_param = args.beta_param
  lambda_param = args.lambda_param

  """ build graph """

  # dropout
  keep_prob = tf.placeholder(tf.float32, name='keep_prob')
  x_in = tf.placeholder(tf.float32, shape=[None, dim_input], name='input_data')
  y_in = tf.placeholder(tf.float32, shape=[None, n_labels], name='output_label')
  c_in = tf.placeholder(tf.float32, shape=[None, n_confounds], name="confounds")

  #TODO: generalize the output types
  loss, neg_x_like, neg_y_like, other_div = na_vib.na_vib(\
    x_in, y_in, c_in,\
    dim_input, dim_out, dim_z,\
    n_hidden_xz, n_hidden_zy,\
    keep_prob,\
    x_output_type=args.x_type, y_output_type=args.y_type,\
    beta_param=beta_param, lambda_param=lambda_param, reuse=False\
  )

  train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

  na_vib_obj = na_vib.classifier(x_in, dim_z, dim_out, n_hidden_xz, n_hidden_zy)

  saver = tf.train.Saver(max_to_keep=None)

  if n_samples < batch_size:
    print("[na_vib] WARNING: n_samples < batch_size")
    batch_size = n_samples
  total_batch = int(n_samples / batch_size)

  val_tracker = validator.Validator()

  with tf.Session() as sess:

    #TODO: modify this!
    if(args.restart_epoch < 0):
      start_epoch = 0
      sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : args.keep_prob})
    else:
      start_epoch = args.restart_epoch + 1
      saver.restore(sess,
        args.param_save_path + args.experiment_name +\
        ("_epoch%i.ckpt" % args.restart_epoch))

    for epoch in range(start_epoch,num_epochs):

      # Random shuffling
      np.random.shuffle(train_total_data)

      #TODO: Modify this to work with train confounder classes
      train_x_shuf = train_total_data[:, :dim_input]
      train_y_shuf = train_total_data[:, dim_input:split_numbers[1]]
      train_c_shuf = train_total_data[:, split_numbers[1]:]

      # Loop over all batches
      for i in range(total_batch):
        # Compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (n_samples)
        batch_x = train_x_shuf[offset:(offset + batch_size), :]
        batch_y = train_y_shuf[offset:(offset + batch_size), :]
        batch_c = train_c_shuf[offset:(offset + batch_size), :]

        _, tot_loss, Lx, Ly, Odiv =\
          sess.run(\
            (train_op, loss, neg_x_like, neg_y_like, other_div),\
            feed_dict={
              x_in: batch_x, \
              y_in: batch_y, \
              c_in: batch_c, \
              keep_prob : args.keep_prob \
            }\
          )

      if args.save_freq > 0 and epoch % args.save_freq == 0:
        saver.save(sess,\
          args.param_save_path + args.experiment_name +\
          ("_epoch%i.ckpt" % epoch))
      #TODO: MAP epoch output

      print("[na_vib] epoch %d: L_tot %03.2f L_x %03.2f L_y %03.2f Div %03.2f"\
        % (epoch, tot_loss, Lx, Ly, Odiv))

      #TODO: MAP epoch output

      if (epoch % 10 == 0 or epoch == num_epochs - 1)\
        and validation_data is not None:
        y_hat = sess.run(na_vib_obj, feed_dict = { x_in: validation_data })
        new_score = \
          np.mean(error_utils.zero_one_abs(validation_labels,np.array(y_hat)))

        print("[na_vib] Validation L: %f" % new_score)
        if val_tracker.new_try(new_score):
          saver.save(sess,\
            args.param_save_path + args.experiment_name + "_MAP.ckpt")
          print("New Record!")
      elif (epoch % 10 == 0 or epoch == num_epochs - 1):
        y_hat = sess.run(na_vib_obj, feed_dict = { x_in: test_data })
        print(error_utils.log_loss(test_labels, np.array(y_hat)))
      #y_hat = np.amax(y_hat,axis=1)
      #error = np.amax(test_labels,axis=1) == y_hat
      #print( sum(np.abs(error)) / np.shape(test_data)[0] )


if __name__ == '__main__':

  # parse arguments
  args = parse_args()
  if args is None:
    exit()

  # main
  main(args)


