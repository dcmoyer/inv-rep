
import tensorflow as tf
import nets
import kl_tools

#TODO:make a generalized encoder

# Gateway
def na_vib(\
  x, y, c,\
  dim_img, dim_out, dim_z,\
  n_hidden_xz, n_hidden_zy, keep_prob,\
  beta_param=1,lambda_param=0, reuse=False,
  x_output_type="zero_one",y_output_type="zero_one",
  name="navib"):

  #
  # encoding
  #
  REDUCTION = tf.losses.Reduction.SUM

  mu, sigma = nets.gaussian_encoder(name,\
    x, n_hidden_xz, dim_z, keep_prob, reuse=reuse)

  # sampling by re-parameterization technique
  z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

  #
  # decoding
  #

  zero_one_flag = False
  scalar_flag = False
  softmax_flag = False
  if y_output_type == "zero_one":
    zero_one_flag = True
  elif y_output_type == "scalar":
    scalar_flag = True
  elif y_output_type == "softmax":
    softmax_flag = True
  else:
    raise("y_output_type %s not recognized, use 'zero_one', 'scalar', or 'softmax'" % y_output_type)

  y_hat = nets.gen_decoder(name +"_zy",
    z,\
    n_hidden_zy, dim_out, keep_prob,\
    output_zero_one=zero_one_flag,\
    output_scalar=scalar_flag,\
    output_softmax=softmax_flag,\
    reuse=reuse\
  )

  # y_hat loss
  if zero_one_flag:
    #y_hat = tf.clip_by_value(y_hat, 1e-8, 1 - 1e-8)
    #y_likelihood = \
    #  tf.reduce_sum(y * tf.log(y_hat) + (1 - y) * tf.log(1 - y_hat), 1)
    #  #-tf.losses.log_loss( labels=y, predictions=y_hat,\
    #  #  reduction=tf.losses.Reduction.MEAN)
    #y_likelihood = tf.reduce_mean(y_likelihood)
    y_likelihood = \
      -tf.losses.absolute_difference( labels=y, predictions=y_hat,\
        reduction=REDUCTION)
  elif softmax_flag:
    y_likelihood = \
      -tf.losses.softmax_cross_entropy( onehot_labels=y, logits=y_hat,\
        reduction=REDUCTION)
  else: 
    y_likelihood = \
      -tf.losses.mean_squared_error(labels=y,predictions=y_hat,\
        reduction=REDUCTION)

  ##
  ## L_x
  ##

  zero_one_flag = False
  scalar_flag = False
  softmax_flag = False
  if x_output_type == "zero_one":
    zero_one_flag = True
  elif x_output_type == "scalar":
    scalar_flag = True
  elif x_output_type == "softmax":
    softmax_flag = True
  else:
    raise("x_output_type %s not recognized, use 'zero_one', 'scalar', or 'softmax'" % y_output_type)

  # passes on the c data
  x_hat = nets.gen_decoder(name + "_zx",
    tf.concat(values=(z,c),axis=1),\
    n_hidden_xz, dim_img, keep_prob,\
    output_zero_one=zero_one_flag,\
    output_scalar=scalar_flag,\
    output_softmax=softmax_flag,\
    reuse=reuse
  )

  # x_hat loss
  if zero_one_flag:
    #x_hat = tf.clip_by_value(x_hat, 1e-8, 1 - 1e-8)
    #x_likelihood = \
    #  tf.reduce_sum(x * tf.log(x_hat) + (1 - x) * tf.log(1 - x_hat), 1)
    #  #-tf.losses.log_loss( labels=x, predictions=x_hat,\
    #  #  reduction=tf.losses.Reduction.MEAN)
    #x_likelihood = tf.reduce_mean(x_likelihood)
    x_likelihood = \
      -tf.losses.absolute_difference( labels=x, predictions=x_hat,\
        reduction=REDUCTION)
  elif scalar_flag:
    x_likelihood = \
      -tf.losses.mean_squared_error(labels=x,predictions=x_hat,\
        reduction=REDUCTION)
  elif softmax_flag:
    x_likelihood = \
      -tf.losses.softmax_cross_entropy( onehot_labels=x, logits=x_hat,\
        reduction=REDUCTION)

  other_div = kl_tools.kl_conditional_and_marg(mu,sigma,dim_z)

  if(lambda_param != 0):
    ELBO = lambda_param*x_likelihood + y_likelihood\
      - (beta_param + lambda_param)*other_div
  else:
    ELBO = y_likelihood - beta_param*other_div

  loss = -ELBO

  return loss, -x_likelihood, -y_likelihood, other_div

def classifier(x, dim_z, dim_out, n_hidden_xz, n_hidden_zy, reuse=True, name="navib",
  y_output_type="zero_one"):

  mu, sigma = nets.gaussian_encoder(name,\
    x, n_hidden_xz, dim_z, 1.0, reuse=reuse)

  # NO sampling, just taking expected/MAP output
  z = mu

  #
  # decoding
  #

  zero_one_flag = False
  scalar_flag = False
  softmax_flag = False
  if y_output_type == "zero_one":
    zero_one_flag = True
  elif y_output_type == "scalar":
    scalar_flag = True
  elif y_output_type == "softmax":
    softmax_flag = True
  else:
    raise("y_output_type %s not recognized, use 'zero_one', 'scalar', or 'softmax'" % y_output_type)

  y_hat = nets.gen_decoder(name +"_zy",
    z,\
    n_hidden_zy, dim_out, 1.0,\
    output_zero_one=zero_one_flag,\
    output_scalar=scalar_flag,\
    output_softmax=softmax_flag,\
    reuse=reuse\
  )

  return y_hat

def encoder(x, dim_z, n_hidden_xz, reuse=False, name="navib"):

  z = nets.gaussian_encoder(name, x, n_hidden_xz, dim_z, 1.0, reuse=reuse)

  return z











