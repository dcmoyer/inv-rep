
import tensorflow as tf
import math



#KL(N_0|N_1) = tr(\sigma_1^{-1} \sigma_0) + 
#  (\mu_1 - \mu_0)\sigma_1^{-1}(\mu_1 - \mu_0) - k +
#  \log( \frac{\det \sigma_1}{\det \sigma_0} )
def all_pairs_gaussian_kl(mu,sigma,dim_z,add_third_term=False):

  sigma_sq = tf.square(sigma) + 1e-8

  #mu is [batchsize x dim_z]
  #sigma is [batchsize x dim_z]

  sigma_sq_inv = tf.reciprocal(sigma_sq)
  #sigma_inv is [batchsize x sizeof(latent_space)]

  #
  # first term
  #

  #dot product of all sigma_inv vectors with sigma
  #is the same as a matrix mult of diag
  first_term = tf.matmul(sigma_sq,tf.transpose(sigma_sq_inv))

  #
  # second term
  #

  #REMEMBER THAT THIS IS SIGMA_1, not SIGMA_0

  r = tf.matmul(mu * mu,tf.transpose(sigma_sq_inv))
  #r is now [batchsize x batchsize] = sum(mu[:,i]**2 / Sigma[j])

  r2 = mu * mu * sigma_sq_inv 
  r2 = tf.reduce_sum(r2,1)
  #r2 is now [batchsize, 1] = mu[j]**2 / Sigma[j]

  #squared distance
  #(mu[i] - mu[j])\sigma_inv(mu[i] - mu[j]) = r[i] - 2*mu[i]*mu[j] + r[j]
  #uses broadcasting
  second_term = 2*tf.matmul(mu, tf.transpose(mu*sigma_sq_inv))
  second_term = r - second_term + tf.transpose(r2)

  ##uncomment to check using tf_tester
  #return second_term

  #
  # third term
  #

  # log det A = tr log A
  # log \frac{ det \Sigma_1 }{ det \Sigma_0 } =
  #   \tr\log \Sigma_1 - \tr\log \Sigma_0 
  # for each sample, we have B comparisons to B other samples...
  #   so this cancels out

  if(add_third_term):
    r = tf.reduce_sum(tf.log(sigma_sq),1)
    r = tf.reshape(r,[-1,1])
    third_term = r - tf.transpose(r)
  else:
    third_term = 0

  #combine and return
  return 0.5 * ( first_term + second_term + third_term - dim_z )


#
# kl_conditional_and_marg
#   \sum_{x'} KL[ q(z|x) \| q(z|x') ] + (B-1) H[q(z|x)]
#

def kl_conditional_and_marg(mu,sigma,dim_z):
  gaussian_ent = \
    0.5 * tf.reduce_sum(dim_z*tf.log(2 * math.pi * math.e) + tf.log( tf.square(sigma) + 1e-8),1)
  #  0.5 * tf.log(\
  #    tf.reduce_prod(2 * math.pi * math.e * tf.square(sigma) + 1e-8,0)\
  #  )

  B = tf.to_float(tf.shape(mu)[0])
  all_pairs_GKL = all_pairs_gaussian_kl(mu,sigma,dim_z,True)
  return tf.reduce_mean(all_pairs_GKL)
  #  * tf.reduce_mean(\
  #  ( + (B - 1) * tf.transpose(gaussian_ent)) - tf.log( B )\
  #) #- log(B)
  #return (1.0 / B) * tf.reduce_mean(\
  #  (tf.reduce_sum(all_pairs_GKL,1) + (B - 1) * tf.transpose(gaussian_ent)) - tf.log( B )\
  #) #- log(B)
#(tf.reduce_sum(all_pairs_GKL,1) + (B - 1)*gaussian_ent)\


