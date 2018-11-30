import tensorflow as tf
import numpy as np
import os
#import vae
import ffd
import glob
import csv

import error_utils

import joblib

import argparse

"""parsing and configuration"""
def parse_args():
  desc = "Tensorflow implementation of 3 layer ff adversary"
  parser = argparse.ArgumentParser(description=desc)

  #
  # negative_anchors
  #

  parser.add_argument('--experiment_name', default=None,
    help='save prefix for params')

  parser.add_argument('--restart_epoch', type=int, default=-1,
    help='epoch to start at, uses param_save_path/experiment_name_#.ckpt')

  parser.add_argument('--latent_and_label_data_path',default=None,\
    help='location of augmented train data (split dependency, sorry =.= )')

  parser.add_argument('--baseline', default=False, action="store_true",
    help='switches into baseline mode')

  parser.add_argument('--save_freq', type=int,default=5,
    help='how often should we save parameters to file? (in epochs)')
  parser.add_argument('--save_freq_adv', type=int,default=5,
    help='how often should we save parameters to file? (in epochs)')

  parser.add_argument('--max_target_epoch',default=None,type=int,\
    help="na_vae or na_vib epoch to test")

  parser.add_argument('--exp_override', default=False, action="store_true",
    help='overrides safety switch and writes over experiment files')

  parser.add_argument('--eval_output', default=None,
    help='file to store the Adv Error (train, val, test) in')

  parser.add_argument('--c_type', default="zero_one",
    help='defines error metric')


  #
  # vae
  #

  parser.add_argument('--dim_z', type=int, default='20',\
    help='Dimension of latent vector')

  parser.add_argument('--non_gauss', default=False, action="store_true",
    help='Sets the dim interpretation to not be a gaussian layer (for non-VAE).')

  parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')

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

  # --augmented_data_path
  if args.latent_and_label_data_path is None:
    print("--latent_and_label_data_path required. Use src/augment_original_data.py to generate.")
    return None

  if os.path.isfile(args.eval_output) and not args.exp_override:
    print("Experiment params already exist! Either restart or use --exp_override flag.")
    return None
  elif args.exp_override:
    os.remove(args.eval_output)
    
  # --dim-z
  try:
      assert (args.dim_z > 0 or args.baseline)
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

  if args.c_type == "zero_one":
    error_func = error_utils.zero_one_abs
    ffd_output_type = "zero_one"
  elif args.c_type == "softmax":
    raise("not implemented yet")
    ffd_output_type = "softmax"
  else:
    error_func = error_utils.rmse
    ffd_output_type = "scalar"

  if args.baseline:
    args.max_target_epoch = 1

  for target_epoch in range(0,args.max_target_epoch,args.save_freq):

    tf.reset_default_graph()

    """ parameters """

    n_hidden = args.n_hidden

    if args.non_gauss:
      dim_z = args.dim_z
    else:
      dim_z = args.dim_z * 2

    """ prepare MNIST data """

    #train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()

    if args.baseline:
      train_total_data, split_numbers, train_size,\
      val_z, _, val_c,\
      test_z, _, test_c = \
        joblib.load( args.latent_and_label_data_path )

      train_z = train_total_data[:,:split_numbers[0]]
      train_c = train_total_data[:,split_numbers[1]:]

      dim_z = split_numbers[0]

    else:
      filename = args.latent_and_label_data_path +\
        ("train/%s_epoch%i_z_and_c.z" % (args.experiment_name, target_epoch))
      train_z, train_c = \
        joblib.load( filename )

      filename = args.latent_and_label_data_path +\
        ("val/%s_epoch%i_z_and_c.z" % (args.experiment_name, target_epoch))
      val_z, val_c = \
        joblib.load( filename )

      filename = args.latent_and_label_data_path +\
        ("test/%s_epoch%i_z_and_c.z" % (args.experiment_name, target_epoch))
      test_z, test_c = \
        joblib.load( filename )

    n_samples = np.shape(train_z)[0]
    n_confounds = np.shape(train_c)[1]

    """ build graph """

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
    c_in = tf.placeholder(tf.float32, shape=[None, n_confounds], name="plmr_confounds")


    ##
    ##  THREE ADVERSARIES!
    ##

    loss0 = ffd.ffd(z_in, c_in, [], n_confounds, keep_prob,\
      ffd_output_type, True, name="L0")
    loss1 = ffd.ffd(z_in, c_in, [n_hidden], n_confounds, keep_prob,\
      ffd_output_type, True, name="L1")
    loss2 = ffd.ffd(z_in, c_in, 2*[n_hidden], n_confounds, keep_prob,\
      ffd_output_type, True, name="L2")
    loss3 = ffd.ffd(z_in, c_in, 3*[n_hidden], n_confounds, keep_prob,\
      ffd_output_type, True, name="L3")

    # optimization
    train_op0 = tf.train.AdamOptimizer(args.learn_rate).minimize(loss0)
    train_op1 = tf.train.AdamOptimizer(args.learn_rate).minimize(loss1)
    train_op2 = tf.train.AdamOptimizer(args.learn_rate).minimize(loss2)
    train_op3 = tf.train.AdamOptimizer(args.learn_rate).minimize(loss3)

    ffd_obj0 = ffd.gen_ffd_logit("L0",\
      z_in, [], n_confounds,1.0,reuse=True) 
    ffd_obj1 = ffd.gen_ffd_logit("L1",\
      z_in, 1*[n_hidden], n_confounds,1.0,reuse=True) 
    ffd_obj2 = ffd.gen_ffd_logit("L2",\
      z_in, 2*[n_hidden], n_confounds,1.0,reuse=True) 
    ffd_obj3 = ffd.gen_ffd_logit("L3",\
      z_in, 3*[n_hidden], n_confounds,1.0,reuse=True) 
    #ffd_obj = ffd.feed_forward_discriminator_logit(\
    #  z_in, n_hidden, n_confounds,1.0,reuse=True) 

    saver = tf.train.Saver(max_to_keep=None)

    train_data = np.concatenate((train_z,train_c),axis=1)

    batch_size = args.batch_size
    if n_samples < batch_size:
      print("[adv] WARNING: n_samples < batch_size")
      batch_size = n_samples
    total_batch = int(n_samples / batch_size)

    with tf.Session() as sess:

      if(args.restart_epoch < 0):
        start_epoch = 0
        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})
      else:
        raise("restart epoch removed")

      for epoch in range(start_epoch,args.num_epochs):

        # Random shuffling
        np.random.shuffle(train_data)

        #TODO: Modify this to work with train confounder classes
        #train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]
        train_z_shuf = train_data[:, :np.shape(train_z)[1]]
        #train_labels_ = train_total_data[:, split_numbers[0]:split_numbers[1]]
        train_c_shuf = train_data[:, np.shape(train_z)[1]:]

        # Loop over all batches
        for i in range(total_batch):
          # Compute the offset of the current minibatch in the data.
          offset = (i * args.batch_size) % (n_samples)
          batch_z = train_z_shuf[offset:(offset + args.batch_size), :]
          batch_c = train_c_shuf[offset:(offset + args.batch_size), :]

          _, tot_loss0 =\
            sess.run( (train_op0, loss0),\
              feed_dict={ z_in: batch_z, c_in: batch_c, keep_prob : 0.9 }
            )

          _, tot_loss1 =\
            sess.run( (train_op1, loss1),\
              feed_dict={ z_in: batch_z, c_in: batch_c, keep_prob : 0.9 }
            )

          _, tot_loss2 =\
            sess.run( (train_op2, loss2),\
              feed_dict={ z_in: batch_z, c_in: batch_c, keep_prob : 0.9 }
            )

          _, tot_loss3 =\
            sess.run( (train_op3, loss3),\
              feed_dict={ z_in: batch_z, c_in: batch_c, keep_prob : 0.9 }
            )

        print("[adv] target_epoch %d epoch %d: L0 %03.2f L1 %03.2f L2 %03.2f L3 %03.2f" %\
          (target_epoch, epoch, tot_loss0, tot_loss1, tot_loss2, tot_loss3))

        if epoch % args.save_freq_adv == 0:
          train_ae = []
          val_ae = []
          test_ae = []
          for ffd_obj in [ffd_obj0, ffd_obj1, ffd_obj2, ffd_obj3]:
            c_hat = sess.run(ffd_obj,feed_dict={z_in : train_z})
            train_ae.append(error_func(train_c,c_hat))

            if val_z is not None:
              c_hat = sess.run(ffd_obj,feed_dict={z_in : val_z})
              val_ae.append(error_func(val_c,c_hat))
            else:
              val_ae.append("NA")

            c_hat = sess.run(ffd_obj,feed_dict={z_in : test_z})
            test_ae.append(error_func(test_c,c_hat))

          with open(args.eval_output,"a") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([target_epoch, epoch] + train_ae + val_ae + test_ae)

  return


if __name__ == '__main__':

  # parse arguments
  args = parse_args()
  if args is None:
    exit()

  # main
  main(args)











