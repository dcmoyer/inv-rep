
import tensorflow as tf
import layers

#here, n_hidden is a list, where each entry makes a layer of that size
def gaussian_encoder(name, x, n_hidden, n_output, keep_prob, reuse=False):

  with tf.variable_scope("%s_gaussian_encoder" % name, reuse=reuse):

    if type(n_hidden) is int:
      n_hidden = [x.get_shape()[1], n_hidden, 2*n_output]
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
        "enc_l%i" % i, reuse, True, keep_prob)

    # The mean parameter is unconstrained
    mean = h[:, :n_output]
    # The standard deviation must be positive. Parametrize with a softplus and
    # add a small epsilon for numerical stability
    stddev = 1e-6 + tf.nn.softplus(h[:, n_output:])

  return mean, stddev

#here, n_hidden is a list, where each entry makes a layer of that size
def gen_encoder(name, x, n_hidden, n_output, keep_prob, reuse=False):

  with tf.variable_scope("%s_gen_encoder" % name, reuse=reuse):

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
        "enc_l%i" % i, reuse, True, keep_prob)

  return h


def gen_decoder(name, z, n_hidden, n_output, keep_prob, reuse=False,\
  output_zero_one=True,output_scalar=False,output_softmax=False):

  with tf.variable_scope("%s_bernoulli_decoder" % name, reuse=reuse):
    if type(n_hidden) is int:
      n_hidden = [z.get_shape()[1], n_hidden]
    elif type(n_hidden) is list:
      n_hidden = [z.get_shape()[1]] + n_hidden
    else:
      raise("type of n_hidden needs to be int or list")

    num_layers = len(n_hidden)

    #h = layers.tanh_layer(x, x.get_shape()[0], n_hidden[0],
    #  "enc_l0", reuse, True, keep_prob)

    h = z
    for i in range(num_layers-1):
      h = layers.tanh_layer(h, n_hidden[i], n_hidden[i+1],
        "dec_l%i" % i, reuse, True, keep_prob)

    if output_zero_one:
      h = layers.sigmoid_layer(h, n_hidden[-1], n_output,
        "dec_l%i" % (num_layers - 1), reuse, True, keep_prob)
    elif output_scalar:
      h = layers.linear_layer(h, n_hidden[-1], n_output,
        "dec_l%i" % (num_layers - 1), reuse, True, keep_prob)
    elif output_softmax:
      h = layers.softmax_layer(h, n_hidden[-1], n_output,
        "dec_l%i" % (num_layers - 1), reuse, True, keep_prob)
    else:
      print("Not reachable")
      raise("wtf")

  return h

##
## probably not useful
##
#def gen_decoder_c(name, z, n_hidden, n_output, keep_prob, c=None,\
#  reuse=False,\
#  output_zero_one=True, output_scalar=False, output_softmax=False):
#
#  if c is None:
#    raise("c_bernoulli called with c None")
#
#  with tf.variable_scope("%s_bernoulli_decoder_c" % name, reuse=reuse):
#    if type(n_hidden) is int:
#      n_hidden = [z.get_shape()[1] + c.get_shape()[1], n_hidden]
#    elif type(n_hidden) is list:
#      n_hidden = [z.get_shape()[1] + c.get_shape()[1]] + n_hidden
#    else:
#      raise("type of n_hidden needs to be int or list")
#
#    num_layers = len(n_hidden)
#
#    #h = layers.tanh_layer(x, x.get_shape()[0], n_hidden[0],
#    #  "enc_l0", reuse, True, keep_prob)
#
#    h = tf.concat(values=(z,c),axis=1)
#    for i in range(num_layers-1):
#      h = layers.tanh_layer(h, n_hidden[i], n_hidden[i+1],
#        "dec_l%i" % i, reuse, True, keep_prob)
#
#    if output_zero_one:
#      h = layers.sigmoid_layer(h, n_hidden[-1], n_output,
#        "dec_l%i" % (num_layers - 1), reuse, True, keep_prob)
#    elif output_scalar:
#      h = layers.linear_layer(h, n_hidden[-1], n_output,
#        "dec_l%i" % (num_layers - 1), reuse, True, keep_prob)
#    elif output_softmax:
#      h = layers.softmax_layer(h, n_hidden[-1], n_output,
#        "dec_l%i" % (num_layers - 1), reuse, True, keep_prob)
#    else:
#      print("Not reachable")
#      raise("wtf")
#
#  return h





