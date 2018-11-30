
import tensorflow as tf
import math
import layers

def feed_forward_discriminator_logit(z, n_hidden,n_output, keep_prob, reuse=False):

  with tf.variable_scope("ff_discriminator_logit",reuse=reuse):
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.0)

    # 1st hidden layer
    w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
    b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
    h0 = tf.matmul(z, w0) + b0
    h0 = tf.nn.tanh(h0)
    h0 = tf.nn.dropout(h0, keep_prob)

    # 2nd hidden layer
    w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
    b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
    h1 = tf.matmul(h0, w1) + b1
    h1 = tf.nn.tanh(h1)
    h1 = tf.nn.dropout(h1, keep_prob)

    # 2nd hidden layer
    w2 = tf.get_variable('w2', [h1.get_shape()[1], n_hidden], initializer=w_init)
    b2 = tf.get_variable('b2', [n_hidden], initializer=b_init)
    h2 = tf.matmul(h1, w2) + b2
    h2 = tf.nn.tanh(h2)
    h2 = tf.nn.dropout(h2, keep_prob)

    # output layer
    # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
    w_out = tf.get_variable('w_out', [h2.get_shape()[1], n_output], initializer=w_init)
    b_out = tf.get_variable('b_out', [n_output], initializer=b_init)

    output = tf.nn.sigmoid( tf.matmul(h2,w_out) + b_out )

  return output

#here, n_hidden is a list, where each entry makes a layer of that size
def gen_ffd_logit(name, x, n_hidden, n_output, keep_prob, reuse=False):

  with tf.variable_scope("%s_gen_ffd_logit" % name, reuse=reuse):

    if type(n_hidden) is int:
      n_hidden = [x.get_shape()[1], n_hidden, n_output]
    elif type(n_hidden) is list:
      n_hidden.append(2*n_output)
      n_hidden = [x.get_shape()[1]] + n_hidden
    else:
      raise("type of n_hidden needs to be int or list")

    num_layers = len(n_hidden)

    #h = layers.tanh_layer(x, x.get_shape()[0], n_hidden[0],
    #  "enc_l0", reuse, True, keep_prob)

    h = x
    for i in range(num_layers-1):
      h = layers.tanh_layer(h, n_hidden[i], n_hidden[i+1],
        "gen_ffd_logit_l%i" % i, reuse, True, keep_prob)

    #output layer
    w_out = tf.get_variable('w_out', [n_hidden[-1], n_output],\
      initializer=layers.INIT_W)
    b_out = tf.get_variable('b_out', [n_output], initializer=layers.INIT_B)

    output = tf.nn.sigmoid( tf.matmul(h,w_out) + b_out )
  return output




def ffd(z, c, n_hidden, n_output, keep_prob, output_type=None,\
  logit=True, reuse=False, name="ffd"):

  if logit :
    #c_hat = feed_forward_discriminator_logit(z,n_hidden,n_output,keep_prob, reuse)
    c_hat = gen_ffd_logit(name, z,n_hidden,n_output,keep_prob, reuse)
  else:
    print("Non-logit c adv not implmented yet")
    exit(1)

  if output_type == "soft_max":
    loss = tf.reduce_mean(\
      tf.nn.softmax_cross_entropy_with_logits( labels=c, logits=c_hat )\
    )
  elif output_type == "zero_one":
    loss = \
      tf.losses.absolute_difference( labels=c, predictions=c_hat,\
        reduction=tf.losses.Reduction.MEAN)
    #loss = \
    #  tf.losses.log_loss( labels=c, predictions=c_hat,\
    #    reduction=tf.losses.Reduction.MEAN)
  elif output_type == "scalar":
    loss = \
      tf.losses.mean_squared_error( labels=c, predictions=c_hat,\
        reduction=tf.losses.Reduction.MEAN)
  else:
    raise("Bad output_type specified")

  return loss



if __name__ == "__main__":

  print(" TESTING ")
  print(" TESTING ")
  print(" TESTING ")
  print(" TESTING ")
  print(" TESTING ")
  print(" TESTING ")

  import numpy as np
  import mnist_data

  """ parameters """

  n_hidden = 64
  dim_img = 28**2  # number of pixels for a MNIST image
  batch_size = 128
  learn_rate = 0.001
  num_epochs = 20

  """ prepare MNIST data """

  train_data, train_size, _, _, test_data, test_labels =\
    mnist_data.prepare_MNIST_data()

  n_samples = train_size 

  """ build graph """

  # dropout
  keep_prob = tf.placeholder(tf.float32, name='keep_prob')
  z_in = tf.placeholder(tf.float32, shape=[None, dim_img], name='latent_variable')
  c_in = tf.placeholder(tf.float32, shape=[None, 10], name="plmr_confounds")

  loss = ffd(z_in, c_in, n_hidden, 10, keep_prob, True)
  train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

  ffd_obj = feed_forward_discriminator_logit(z_in, n_hidden, 10,1.0, True) 

  saver = tf.train.Saver(max_to_keep=None)

  total_batch = int(n_samples / batch_size)

  with tf.Session() as sess:

    start_epoch = 0
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

    for epoch in range(start_epoch,num_epochs):

      # Random shuffling
      np.random.shuffle(train_data)

      #TODO: Modify this to work with train confounder classes
      #train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]
      train_z_shuf = train_data[:, :dim_img]
      #train_labels_ = train_total_data[:, split_numbers[0]:split_numbers[1]]
      train_c_shuf = train_data[:, dim_img:]

      # Loop over all batches
      for i in range(total_batch):
        # Compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (n_samples)
        batch_z = train_z_shuf[offset:(offset + batch_size), :]
        batch_c = train_c_shuf[offset:(offset + batch_size), :]

        _, tot_loss =\
          sess.run(\
            (train_op, loss),\
            feed_dict={
              z_in: batch_z, \
              c_in: batch_c,\
              keep_prob : 0.9, \
            }\
          )

      print("[adv][test] epoch %d: L_tot %03.2f" % (epoch, tot_loss))

      #TODO: MAP epoch output

    c_hat = sess.run(ffd_obj, feed_dict = { z_in: test_data })
    c_hat = np.amax(c_hat,axis=1)
    error = np.amax(test_labels,axis=1) == c_hat
    print( sum(np.abs(error)) / np.shape(test_data)[0] )























