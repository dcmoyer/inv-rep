
import tensorflow as tf

INIT_W = tf.contrib.layers.variance_scaling_initializer()
INIT_B = tf.constant_initializer(0.)


def tanh_layer(input, in_dim, out_dim, name, reuse,\
  dropout=True, keep_prob=1.0):
  with tf.variable_scope(name,reuse=reuse):
    w = tf.get_variable('w', [in_dim, out_dim], initializer=INIT_W)
    b = tf.get_variable('b', [out_dim], initializer=INIT_B)
    h = tf.matmul(input, w) + b
    h = tf.nn.tanh(h)

    if(dropout):
      tf.nn.dropout(h,keep_prob)

  return h

def sigmoid_layer(input, in_dim, out_dim, name, reuse,\
  dropout=True, keep_prob=1.0):
  with tf.variable_scope(name,reuse=reuse):
    w = tf.get_variable('w', [in_dim, out_dim], initializer=INIT_W)
    b = tf.get_variable('b', [out_dim], initializer=INIT_B)
    h = tf.matmul(input, w) + b
    h = tf.nn.sigmoid(h)

    if(dropout):
      tf.nn.dropout(h,keep_prob)

  return h

def linear_layer(input, in_dim, out_dim, name, reuse,\
  dropout=True, keep_prob=1.0):
  with tf.variable_scope(name,reuse=reuse):
    w = tf.get_variable('w', [in_dim, out_dim], initializer=INIT_W)
    b = tf.get_variable('b', [out_dim], initializer=INIT_B)
    h = tf.matmul(input, w) + b

    if(dropout):
      tf.nn.dropout(h,keep_prob)

  return h

def softmax_layer(input, in_dim, out_dim, name, reuse,\
  dropout=True, keep_prob=1.0):
  with tf.variable_scope(name,reuse=reuse):
    w = tf.get_variable('w', [in_dim, out_dim], initializer=INIT_W)
    b = tf.get_variable('b', [out_dim], initializer=INIT_B)
    h = tf.matmul(input, w) + b
    h = tf.nn.softmax(h)

    if(dropout):
      tf.nn.dropout(h,keep_prob)

  return h


