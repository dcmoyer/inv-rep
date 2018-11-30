import numpy as np


def log_loss(c,c_hat):
  c_hat = np.clip(c_hat,1e-8,1-1e-8)
  return -c * np.log(c_hat) - (1 - c)*np.log(1 - c_hat)

def zero_one_thold(c,c_hat,thold=0.5):
  c_hat[c_hat < thold] = 0
  c_hat[c_hat >= thold] = 1
  return np.asscalar(sum((c_hat == c).astype(int)) / c.shape[0])

def zero_one_abs(c,c_hat):
  return np.asscalar(np.sum(np.abs(c_hat - c)) / c.shape[0])

def rmse(c,c_hat):
  return np.mean((c - c_hat)*(c - c_hat))


