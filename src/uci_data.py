
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy as np
import pandas as pd

from six.moves import urllib

import tensorflow as tf

ADULT_SOURCE_URL =\
  'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
GERMAN_SOURCE_URL =\
  'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/'

ADULT_RAW_COL_NAMES =\
  ["age",\
    "workclass",\
    "fnlwgt",\
    "education",\
    "education-num",\
    "marital-status",\
    "occupation",\
    "relationship",\
    "race",\
    "sex",\
    "capital-gain",\
    "capital-loss",\
    "hours-per-week",\
    "native-country",
    "income"\
  ]

ADULT_RAW_COL_FACTOR =\
  [1,3,5,6,7,8,13]

ADULT_VALIDATION_SPLIT = 0.5

GERMAN_RAW_COL_NAMES =\
  ["checking_acc",\
    "duration",\
    "credit_hist",\
    "purpose",\
    "credit_amount",\
    "savings",\
    "employment_status",\
    "install_rate",\
    "relationship_and_sex",\
    "debtors",\
    "res_interval",\
    "property",\
    "age",\
    "other_plans",
    "housing",\
    "credits_at_bank",
    "job",
    "liable_persons",
    "phone",
    "foreign",
    "cr_good_bad"
  ]

GERMAN_RAW_COL_FACTOR =\
  [0,2,3,5,6,8,9,11,13,14,16,18,19]

GERMAN_VAL_SPLIT = 0.2
GERMAN_TEST_SPLIT = 0.2

DATA_DIRECTORY = "data/"

def maybe_download(adult_flag=False, german_flag=False):
  if not adult_flag and not german_flag:
    raise("Neither flag specified, aborting.")

  if adult_flag:
    maybe_download_adult("adult.data")
    maybe_download_adult("adult.test")

  if german_flag:
    maybe_download_german("german.data")

# Download ADULT data
def maybe_download_adult(filename):

  if not tf.gfile.Exists(DATA_DIRECTORY):
    tf.gfile.MakeDirs(DATA_DIRECTORY + "raw")

  filepath = os.path.join(DATA_DIRECTORY + "raw", filename)

  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(ADULT_SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')

  return filepath

# Download german data
def maybe_download_german(filename):

  if not tf.gfile.Exists(DATA_DIRECTORY):
    tf.gfile.MakeDirs(DATA_DIRECTORY + "raw")

  filepath = os.path.join(DATA_DIRECTORY + "raw", filename)

  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(GERMAN_SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')

  return filepath


def process_adult_data():
  #train_data = np.loadtxt(os.path.join(DATA_DIRECTORY + "raw","adult.data"))
  train_data = pd.read_table(\
    os.path.join(DATA_DIRECTORY + "raw","adult.data"),\
    delimiter=", ", header=None, names=ADULT_RAW_COL_NAMES,
    na_values="?",keep_default_na=False)
  test_data = pd.read_table(\
    os.path.join(DATA_DIRECTORY + "raw","adult.test"),\
    delimiter=", ", header=None, names=ADULT_RAW_COL_NAMES,
    na_values="?",keep_default_na=False, skiprows=1)

  train_data.dropna(inplace=True)
  test_data.dropna(inplace=True)

  all_data = pd.concat([train_data,test_data])

  all_data = pd.get_dummies(all_data,\
    columns=[ADULT_RAW_COL_NAMES[i] for i in ADULT_RAW_COL_FACTOR])

  all_data.loc[all_data.income == ">50K","income"] = 1
  all_data.loc[all_data.income == ">50K.","income"] = 1
  all_data.loc[all_data.income == "<=50K","income"] = 0
  all_data.loc[all_data.income == "<=50K.","income"] = 0

  all_data.loc[all_data.sex == "Female","sex"] = 1
  all_data.loc[all_data.sex == "Male","sex"] = 0

  #all_data = pd.get_dummies(all_data,columns=["income"],drop_first=True)

  cutoff = train_data.shape[0]
  train_data = all_data.loc[:cutoff,\
    (all_data.columns != "income") & (all_data.columns != "sex")]
  train_c = all_data.loc[:cutoff,all_data.columns == "sex"]
  train_labels = all_data.loc[:cutoff,all_data.columns == "income"]

  test_data = all_data.loc[cutoff:,\
    (all_data.columns != "income") & (all_data.columns != "sex")]
  test_c = all_data.loc[cutoff:,all_data.columns == "sex"]

  test_labels = all_data.loc[cutoff:,all_data.columns == "income"]

  col_valid_in_train_data =\
     [len(train_data.loc[:,x].unique()) > 1 for x in train_data.columns]
  col_valid_in_test_data =\
     [len(test_data.loc[:,x].unique()) > 1 for x in test_data.columns]

  col_valid = list(map(lambda x,y: x and y, col_valid_in_train_data, col_valid_in_test_data))

  train_data = train_data.loc[:,col_valid]
  test_data = test_data.loc[:,col_valid]

  #split off validation data
  cutoff = int(ADULT_VALIDATION_SPLIT * train_data.shape[0])
  val_data = train_data.loc[cutoff:,:]
  train_data = train_data.loc[:cutoff,:]

  val_labels = train_labels.loc[cutoff:,:]
  train_labels = train_labels.loc[:cutoff,:]

  val_c = train_c.loc[cutoff:,:]
  train_c = train_c.loc[:cutoff,:]

  split_numbers = (train_data.shape[1],train_data.shape[1] + 1)
  train_size = train_data.shape[0]

  #data normalization
  maxes = np.maximum(np.maximum(\
    train_data.max(axis=0),\
    val_data.max(axis=0)),\
    test_data.max(axis=0))

  train_data = train_data / maxes
  test_data = test_data / maxes
  val_data = val_data / maxes

  train_data = np.concatenate(\
    (train_data.as_matrix(), train_labels.as_matrix(), train_c.as_matrix()),\
    axis=1\
  )

  return train_data, split_numbers, train_size,\
    val_data.as_matrix(), val_labels.as_matrix(), val_c.as_matrix(),\
    test_data.as_matrix(), test_labels.as_matrix(), test_c.as_matrix()


def process_german_data():
  #train_data = np.loadtxt(os.path.join(DATA_DIRECTORY + "raw","adult.data"))
  all_data = pd.read_table(\
    os.path.join(DATA_DIRECTORY + "raw","german.data"),\
    delimiter=" ", header=None, names=GERMAN_RAW_COL_NAMES,
    na_values="?",keep_default_na=False)

  all_data.dropna(inplace=True)

  all_data = all_data.assign(sex=(all_data.relationship_and_sex == "A92").astype(int) | \
    (all_data.relationship_and_sex == "A95").astype(int))
  col_names = GERMAN_RAW_COL_NAMES +["sex"] 
  all_data.loc[:,all_data.columns == "cr_good_bad"] =\
    all_data.loc[:,all_data.columns == "cr_good_bad"] - 1

  all_data = pd.get_dummies(all_data,\
    columns=[col_names[i] for i in GERMAN_RAW_COL_FACTOR])

  #all_data = pd.get_dummies(all_data,columns=["income"],drop_first=True)

  cutoff = int(all_data.shape[0] * (1.0 - GERMAN_TEST_SPLIT))
  train_data = all_data.loc[:cutoff,\
    (all_data.columns != "cr_good_bad") & (all_data.columns != "sex")]
  train_c = all_data.loc[:cutoff,all_data.columns == "sex"]
  train_labels = all_data.loc[:cutoff,all_data.columns == "cr_good_bad"]

  test_data = all_data.loc[cutoff:,\
    (all_data.columns != "cr_good_bad") & (all_data.columns != "sex")]
  test_c = all_data.loc[cutoff:,all_data.columns == "sex"]

  test_labels = all_data.loc[cutoff:,all_data.columns == "cr_good_bad"]

  col_valid_in_train_data =\
     [len(train_data.loc[:,x].unique()) > 1 for x in train_data.columns]
  col_valid_in_test_data =\
     [len(test_data.loc[:,x].unique()) > 1 for x in test_data.columns]

  col_valid = list(map(lambda x,y: x and y, col_valid_in_train_data, col_valid_in_test_data))

  train_data = train_data.loc[:,col_valid]
  test_data = test_data.loc[:,col_valid]

  #split off validation data
  cutoff = int(GERMAN_VAL_SPLIT * train_data.shape[0])
  val_data = train_data.loc[cutoff:,:]
  train_data = train_data.loc[:cutoff,:]

  val_labels = train_labels.loc[cutoff:,:]
  train_labels = train_labels.loc[:cutoff,:]

  val_c = train_c.loc[cutoff:,:]
  train_c = train_c.loc[:cutoff,:]

  split_numbers = (train_data.shape[1],train_data.shape[1] + 1)
  train_size = train_data.shape[0]

  #data normalization
  maxes = np.maximum(np.maximum(\
    train_data.max(axis=0),\
    val_data.max(axis=0)),\
    test_data.max(axis=0))

  train_data = train_data / maxes
  test_data = test_data / maxes
  val_data = val_data / maxes

  train_data = np.concatenate(\
    (train_data.as_matrix(), train_labels.as_matrix(), train_c.as_matrix()),\
    axis=1\
  )

  return train_data, split_numbers, train_size,\
    val_data.as_matrix(), val_labels.as_matrix(), val_c.as_matrix(),\
    test_data.as_matrix(), test_labels.as_matrix(), test_c.as_matrix()

def process_german_data_age_as_c():
  #train_data = np.loadtxt(os.path.join(DATA_DIRECTORY + "raw","adult.data"))
  all_data = pd.read_table(\
    os.path.join(DATA_DIRECTORY + "raw","german.data"),\
    delimiter=" ", header=None, names=GERMAN_RAW_COL_NAMES,
    na_values="?",keep_default_na=False)

  all_data.dropna(inplace=True)

  all_data.loc[:,all_data.columns == "cr_good_bad"] =\
    all_data.loc[:,all_data.columns == "cr_good_bad"] - 1

  all_data.loc[:,all_data.columns == "age"] =\
    all_data.loc[:,all_data.columns == "age"] > 25
  all_data.loc[all_data.age == True,all_data.columns == "age"] = 1
  all_data.loc[all_data.age == False,all_data.columns == "age"] = 0

  col_names = GERMAN_RAW_COL_NAMES
  all_data = pd.get_dummies(all_data,\
    columns=[col_names[i] for i in GERMAN_RAW_COL_FACTOR])

  #all_data = pd.get_dummies(all_data,columns=["income"],drop_first=True)

  cutoff = int(all_data.shape[0] * (1.0 - GERMAN_TEST_SPLIT))
  train_data = all_data.loc[:cutoff,\
    (all_data.columns != "cr_good_bad") & (all_data.columns != "age")]
  train_c = all_data.loc[:cutoff,all_data.columns == "age"]
  train_labels = all_data.loc[:cutoff,all_data.columns == "cr_good_bad"]

  test_data = all_data.loc[cutoff:,\
    (all_data.columns != "cr_good_bad") & (all_data.columns != "age")]
  test_c = all_data.loc[cutoff:,all_data.columns == "age"]

  test_labels = all_data.loc[cutoff:,all_data.columns == "cr_good_bad"]

  col_valid_in_train_data =\
     [len(train_data.loc[:,x].unique()) > 1 for x in train_data.columns]
  col_valid_in_test_data =\
     [len(test_data.loc[:,x].unique()) > 1 for x in test_data.columns]

  col_valid = list(map(lambda x,y: x and y, col_valid_in_train_data, col_valid_in_test_data))

  train_data = train_data.loc[:,col_valid]
  test_data = test_data.loc[:,col_valid]

  #split off validation data
  cutoff = int(GERMAN_VAL_SPLIT * train_data.shape[0])
  val_data = train_data.loc[cutoff:,:]
  train_data = train_data.loc[:cutoff,:]

  val_labels = train_labels.loc[cutoff:,:]
  train_labels = train_labels.loc[:cutoff,:]

  val_c = train_c.loc[cutoff:,:]
  train_c = train_c.loc[:cutoff,:]

  split_numbers = (train_data.shape[1],train_data.shape[1] + 1)
  train_size = train_data.shape[0]

  #data normalization
  maxes = np.maximum(np.maximum(\
    train_data.max(axis=0),\
    val_data.max(axis=0)),\
    test_data.max(axis=0))

  train_data = train_data / maxes
  test_data = test_data / maxes
  val_data = val_data / maxes

  train_data = np.concatenate(\
    (train_data.as_matrix(), train_labels.as_matrix(), train_c.as_matrix()),\
    axis=1\
  )

  return train_data, split_numbers, train_size,\
    val_data.as_matrix(), val_labels.as_matrix(), val_c.as_matrix(),\
    test_data.as_matrix(), test_labels.as_matrix(), test_c.as_matrix()





if __name__ == "__main__":
  maybe_download(adult_flag = True, german_flag = True)

  train_data, split_numbers, train_size,\
    validation_data, validation_labels, validation_c,\
    test_data, test_labels, test_c = \
    process_adult_data()

  import joblib

  #the one in which I'm just tired
  joblib.dump(
    value = ( train_data, split_numbers, train_size,\
      validation_data, validation_labels, validation_c,\
      test_data, test_labels, test_c ),\
    filename = "data/adult_proc.z",\
    compress = ("zlib",1)\
  )

  print("adult train c split %f" % np.asscalar(np.sum(train_data[:,-1])/train_data.shape[0]))
  print("adult train y split %f" % np.asscalar(np.sum(train_data[:,-2])/train_data.shape[0]))
  print("adult test c split %f" % np.asscalar(np.sum(test_c)/test_c.shape[0]))
  print("adult test y split %f" % np.asscalar(np.sum(test_labels)/test_labels.shape[0]))

  train_data, split_numbers, train_size,\
    validation_data, validation_labels, validation_c,\
    test_data, test_labels, test_c = \
    process_german_data()

  #the one in which I'm just tired
  joblib.dump(
    value = ( train_data, split_numbers, train_size,\
      validation_data, validation_labels, validation_c,\
      test_data, test_labels, test_c ),\
    filename = "data/german_proc_gender.z",\
    compress = ("zlib",1)\
  )

  print("german test c split %f" % np.asscalar(np.sum(test_c)/test_c.shape[0]))
  print("german test y split %f" % np.asscalar(np.sum(test_labels)/test_labels.shape[0]))

  train_data, split_numbers, train_size,\
    validation_data, validation_labels, validation_c,\
    test_data, test_labels, test_c = \
    process_german_data_age_as_c()

  #the one in which I'm just tired
  joblib.dump(
    value = ( train_data, split_numbers, train_size,\
      validation_data, validation_labels, validation_c,\
      test_data, test_labels, test_c ),\
    filename = "data/german_proc.z",\
    compress = ("zlib",1)\
  )

  print("german test c split %f" % np.asscalar(np.sum(test_c)/test_c.shape[0]))
  print("german test y split %f" % np.asscalar(np.sum(test_labels)/test_labels.shape[0]))







