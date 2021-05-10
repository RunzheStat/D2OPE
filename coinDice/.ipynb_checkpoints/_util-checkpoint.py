# instance.method = MethodType(method, instance)
# !aws codecommit list-repositories
# !autopep8 --in-place --aggressive --aggressive brac_dual_agent.py

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

#############################################################################
# %matplotlib inline
# !tar -czf data.tar.gz data
# !tar -czf code.tar.gz code
from inspect import getsource
from importlib import reload
from livelossplot import PlotLosses
import pytz
import tensorflow as tf
import silence_tensorflow.auto
# tz_NY = pytz.timezone('America/New_York') 
# dt.now(tz_NY).strftime("%D:%H:%M:%S")

from typing import Dict, List, Set, Tuple
from datetime import datetime as dt
import itertools
import io
import sys
import gym
import ray

import warnings
warnings.simplefilter("ignore")
# from sagemaker import get_execution_role
# role = get_execution_role()
from IPython.display import clear_output

from tqdm import tqdm
# https://github.com/tqdm/tqdm
"""
pbar = tqdm(["a", "b", "c", "d"])
for char in pbar:
    time.sleep(0.25)
    pbar.set_description("Processing %s" % char)
for i in tqdm(range(10)):

"""
def smooth_loss(loss, freq):
    loss = arr(loss).copy()
    return np.mean(loss.reshape(-1, freq), axis = -1)
from types import MethodType
import functools 
from functools import reduce 
#############################################################################
# Packages
import scipy as sp
import pandas as pd
from pandas import DataFrame as DF
# import statsmodels.api as sm # !pip install statsmodels
from matplotlib.pyplot import hist
import pickle
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

####################################

# Random
import random
from random import seed as rseed
from numpy.random import seed as npseed
from numpy import absolute as np_abs
from numpy.random import normal as rnorm
from numpy.random import uniform as runi
from numpy.random import binomial as rbin
from numpy.random import poisson as rpoisson
from numpy.random import shuffle,randn, permutation # randn(d1,d2) is d1*d2 i.i.d N(0,1)
from numpy import squeeze
from numpy.linalg import solve
####################################

# Numpy
import numpy as np
from numpy import mean, var, std, median
from numpy import array as arr
from numpy import sqrt, log, cos, sin, exp, dot, diag, ones, identity, zeros, roll, multiply, stack, concatenate, transpose
from numpy import concatenate as v_add
from numpy.linalg import norm, inv
from numpy import apply_along_axis as apply
from numpy.random import multinomial, choice
####################################

# sklearn
import sklearn as sk
from sklearn import preprocessing as pre
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression as lm
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


from scipy.special import softmax
#############################################################################
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

np.set_printoptions(precision = 4)
#############################################################################
import time
now = time.time
import smtplib, ssl

import datetime, pytz

def EST():
    return datetime.datetime.now().astimezone(pytz.timezone('US/Eastern')).strftime("%H:%M, %m/%d")

#############################################################################
dash = "--------------------------------------"
DASH = "\n" + "--------------------------------------" + "\n"
Dash = "\n" + dash
dasH = dash + "\n"
#############################################################################
#%% utility funs
from multiprocessing import Pool
import multiprocessing
n_cores = multiprocessing.cpu_count()



def mute():
    sys.stdout = open(os.devnull, 'w')    

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))
        
def parmap(f, X, nprocs = multiprocessing.cpu_count(), **args):#-2
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()
    
    def g(x):
        return f(x, **args)
    
    proc = [multiprocessing.Process(target=fun, args=(g, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]

def setminus(A, B):
    return [item for item in A if item not in B]

def listinlist2list(theList):
    return [item for sublist in theList for item in sublist]

def if_exist(obj):
    return obj in locals() or obj in globals()

def getSize(one_object):
    print(one_object.memory_usage().sum() / 1024 ** 2, "MB")
#     print(sys.getsizeof(one_object) // 1024, "MB")

def dump(file, path):
    pickle.dump(file, open(path, "wb"))
    
def load(path):
    return pickle.load(open(path, "rb"))

def get_MB(a):
    MB = sys.getsizeof(a) / 1024 / 1024
    return MB

def hstack_all_comb(array1, array2):
    # array1 is on the left and also changes faster
    res = np.hstack([
        np.tile(array1, (array2.shape[0], 1))
    , np.repeat(array2, array1.shape[0], axis=0)]
    )
    return res


def quantile(a, p):
    r = [a[0] for a in DF(a).quantile(p).values]
    return np.round(r, 3)

def flatten(l): 
    # list of sublist -> list
    return [item for sublist in l for item in sublist]

def change_rate(old_targets, new_targets, numpy = False):
    if numpy:
        diff = np.mean(abs(new_targets-old_targets)) / (np.mean(abs(old_targets))+1e-6)
    else:
        diff = abs(new_targets-old_targets).mean() / (abs(old_targets).mean()+1e-6)
    return min(1.0, diff)



#############################################################################

# pd.options.display.max_rows = 10

# with open('pred_columns.txt', 'w') as filehandle:
#     k = 0
#     for listitem in list(a):
#         filehandle.write('{}    {}\n'.format(k, listitem))
#         k += 1

def print_all(dat, column_only = True):
    if column_only:
        with pd.option_context('display.max_columns', None):  # more options can be specified also
            print(dat)
    else:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(dat)

            
def quantile(a):
    return np.percentile(a, range(0,110,10))

#############################################################################

def unzip(path, zip_type = "tar_gz"):
    if zip_type == "tar_gz":
        import tarfile
        tar = tarfile.open(path, "r:gz")
        tar.extractall()
        tar.close()
    elif zip_type == "zip":        
        from zipfile import ZipFile
        with ZipFile(path, 'r') as zipObj:
           # Extract all the contents of zip file in current directory
           zipObj.extractall()

            
# import shutil

# total, used, free = shutil.disk_usage("/")

# print("Total: %d GiB" % (total // (2**30)))
# print("Used: %d GiB" % (used // (2**30)))
# print("Free: %d GiB" % (free // (2**30)))

#############################################################################

# !pip install termcolor
from termcolor import colored, cprint

# https://pypi.org/project/termcolor/#description
def printR(theStr):
    print(colored(theStr, 'red'))
          
def printG(theStr):
    print(colored(theStr, 'green'))
          
def printB(theStr):
    print(colored(theStr, 'blue'))

    
def sets_intersection(d):
    return list(reduce(set.intersection, [set(item) for item in d ]))

def select_each_row(array, idx):
    return np.take_along_axis(array, idx[:,None], axis=1)

def subtract_each_column(mat, col):
    return (mat.transpose() - col).transpose()

def sample_split(L, N):
    """ replay buffer?
    """
    kf = KFold(n_splits=L)
    kf.get_n_splits(zeros(N))
    split_ind = {}
    k = 0
    for i, j in kf.split(range(N)):
        split_ind[k] = {"train_ind" : i, "test_ind" : j}
        k += 1
    return split_ind

def row_repeat(mat, rep, full_block = False):
    if full_block:
        return np.tile(mat, (rep, 1))
    else:
        return np.repeat(mat, rep, axis=0)
    
def SARS2traj(SARS, S_dim = 3):
    states = arr([sars[0][:S_dim] for sars in SARS])
    actions =  arr([sars[1] for sars in SARS])
    return states, actions