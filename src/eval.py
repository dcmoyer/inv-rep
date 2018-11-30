import tensorflow as tf
import numpy as np
import os

import na_vib
import error_utils 
import glob
import csv

import joblib

import argparse

"""parsing and configuration"""
def parse_args():
  desc = "Tensorflow implementation of 'Variational AutoEncoder (VAE)'"
  parser = argparse.ArgumentParser(description=desc)

  #
  # negative_anchors
  #

  parser.add_argument('--lambda_param', type=float,default=0.0001,
    help='negative anchor trade-off parameter')

  parser.add_argument('--param_save_path', default="params/",
    help='save path for params')

  parser.add_argument('--experiment_name', default=None,
    help='save prefix for params')

  parser.add_argument('--start_epoch', type=int, default=0,
    help='epoch to start at, uses param_save_path/experiment_name_#.ckpt')

  parser.add_argument('--save_freq', type=int,default=5,
    help='how often should we save parameters to file? (in epochs)')

  parser.add_argument('--num_epochs', type=int, default=250,\
    help='The number of epochs to run')

  parser.add_argument('--augmented_data_path',default=None,\
    help='location of augmented data (split dependency, sorry =.= )')

  parser.add_argument('--exp_override', default=False, action="store_true",
    help='overrides safety switch and writes over experiment files')

  parser.add_argument('--output_latent_codes', default=False, action="store_true",
    help='flag to output z to a file')

  parser.add_argument('--output_pred_error', default=False, action="store_true",
    help='flag to output predictive error')

  parser.add_argument('--pred_error_file',default=None,\
    help='file to output prediction error into')

  parser.add_argument('--y_type', default="zero_one",
    help='defines error metric for pred error')

  parser.add_argument('--output_loss_evals', default=False, action="store_true",
    help='flag to output loss to a file')


  #
  # vae
  #

  parser.add_argument('--outputs_path', type=str, default='eval/',
    help='File path of output images')

  parser.add_argument('--outputs_prefix', type=str, default='',
    help='File prefix of output images')

  parser.add_argument('--dim_z', type=int, default='20',\
    help='Dimension of latent vector', required = True)

  parser.add_argument('--n_hidden', type=int, default=500,
    help='Number of hidden units in MLP')

  parser.add_argument('--n_hidden_xz', type=int, default=500,
    help='Number of hidden units in encoder from x to z')

  parser.add_argument('--n_hidden_zy', type=int, default=500,
    help='Number of hidden units in predictor from z to y')

  return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):

  # --experiment_name
  if not args.experiment_name:
    print("Experiment Needs a Name! Aborting.")
    return None

  #
  # TODO: remove bad dir creation
  #

  # --outputs_path
  try:
      os.mkdir(args.outputs_path)
      
      for dataset_name in ["train", "val", "test"]:
        os.mkdir(os.path.join(args.outputs_path,dataset_name))
  except(FileExistsError):
      pass

  # --param_save_path
  try:
      os.mkdir(args.param_save_path)
  except(FileExistsError):
      pass

  # --augmented_data_path
  if args.augmented_data_path is None:
    print("--augmented_data_path required. Use src/uci_data.py to generate.")
    return None

  # --dim-z
  try:
      assert args.dim_z > 0
  except:
      print('dim_z must be positive integer')
      return None

  # --n_hidden
  try:
      assert args.n_hidden >= 1
  except:
      print('number of hidden units must be larger than one')

  return args


def main(args):

  """ parameters """

  n_hidden = args.n_hidden
  dim_z = args.dim_z

  train_total_data, split_numbers, train_size,\
  validation_data, validation_labels, validation_c,\
  test_data, test_labels, test_c = \
    joblib.load( args.augmented_data_path )

  n_samples = train_size
  n_confounds = train_total_data.shape[1] - split_numbers[1]
  dim_input = split_numbers[0]

  """ build graph """

  # input placeholders
  # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
  x_in = tf.placeholder(tf.float32, shape=[None, dim_input], name='input_img')
  x = tf.placeholder(tf.float32, shape=[None, dim_input], name='target_img')

  #generalize to multiple latent c factors
  c = tf.placeholder(tf.float32, shape=[None, n_confounds], name='confounds')

  # dropout
  keep_prob = tf.placeholder(tf.float32, name='keep_prob')

  # input for PMLR
  z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
  c_in = tf.placeholder(tf.float32, shape=[None, n_confounds], name="plmr_confounds")

  # network architecture
  #y, z, loss, neg_marg_likelihood, KL_divergence, other_div =\
  #  na_vae.autoencoder(x_hat, x, dim_input, dim_z, n_hidden, keep_prob,\
  #  args.lambda_param, has_c_term=True, c=c)

  dim_out = split_numbers[1] - split_numbers[0]
  classifier = na_vib.classifier(\
    x_in, dim_z, dim_out, args.n_hidden_xz, args.n_hidden_zy,reuse=False\
  )

  encoder = na_vib.encoder(x_in, dim_z, args.n_hidden_xz, True)

  saver = tf.train.Saver(max_to_keep=None)

  with tf.Session() as sess:

    for epoch in range(args.start_epoch,args.num_epochs,args.save_freq):

      saver.restore(sess,
        args.param_save_path + args.experiment_name +\
        ("_epoch%i.ckpt" % epoch))

      for dataset_name in ["train", "val", "test"]:

        if dataset_name == "val":
          x_eval = validation_data
          c_eval = validation_c
        elif dataset_name == "test":
          x_eval = test_data
          c_eval = test_c
        else:
          x_eval = train_total_data[:, :split_numbers[0]]
          c_eval = train_total_data[:, split_numbers[1]:]

        if args.output_latent_codes:

          if x_eval is not None:
            z_eval_mean,z_eval_sigma = sess.run( encoder,\
              feed_dict = {\
                x_in: x_eval
              }\
            )

            z_eval = np.concatenate((z_eval_mean,z_eval_sigma),axis=1)
          else:
            z_eval = None

          output_file = os.path.join(args.outputs_path, dataset_name,\
            "%s%s_epoch%i_z_and_c.z" %\
              (args.outputs_prefix, args.experiment_name, epoch)\
          )
          joblib.dump(
            value = ( z_eval, c_eval ),\
            filename = output_file,\
            compress = ("zlib",1)\
          )

        if args.output_loss_evals:

          tot_loss, loss_likelihood, loss_div2, loss_div2 =\
            sess.run(\
              (loss, neg_marg_likelihood, KL_divergence, other_div),\
              feed_dict={
                x_in: x_eval, \
                x: x_eval, \
                keep_prob : 1.0, \
                c: c_eval\
              }\
            )

          print(
            ("set %s " % dataset_name ) + 
            ("epoch %d: L_tot %03.2f L_like %03.2f L_div1 %03.2f L_div2 %03.2f"\
              % (epoch, tot_loss, loss_likelihood, loss_div1, loss_div2))\
          )

        if args.output_pred_error:
          if args.y_type == "zero_one":
            error_func = error_utils.zero_one_abs
            ffd_output_type = "zero_one"
          elif args.y_type == "softmax":
            raise("not implemented yet")
            ffd_output_type = "softmax"
          else:
            error_func = error_utils.rmse
            ffd_output_type = "scalar"

          y_hat = sess.run(classifier,\
            feed_dict = { x_in: train_total_data[:,:split_numbers[0]] })
          train_pe = error_func(train_total_data[:,split_numbers[0]:split_numbers[1]],y_hat)

          if validation_data is not None:
            y_hat = sess.run(classifier,\
              feed_dict = { x_in: validation_data })
            val_pe = error_func(validation_labels,y_hat)
          else:
            val_pe = "NA"

          y_hat = sess.run(classifier,\
            feed_dict = { x_in: test_data })
          test_pe = error_func(test_labels,y_hat)

          with open(args.pred_error_file,"a") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([epoch, train_pe, val_pe, test_pe])

  return


if __name__ == '__main__':

  # parse arguments
  args = parse_args()
  if args is None:
    exit()

  # main
  main(args)




