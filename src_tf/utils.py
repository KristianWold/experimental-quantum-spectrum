import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import tensorflow as tf
import pickle
import os
from set_precision import *

from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm
import multiprocessing as mp


def numberToBase(n, b, num_digits):
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b

    while len(digits) < num_digits:
        digits.append(0)
    return digits[::-1]


def kron(*args):
    length = len(args)
    A = args[0]
    for i in range(1, length):
        A = tf.experimental.numpy.kron(A, args[i])

    return A


def tf_kron(A, B):
    if len(A.shape) == 2:
        A = tf.expand_dims(A, axis=0)
        B = tf.expand_dims(B, axis=0)

    batch = A.shape[0]
    dim_a1 = A.shape[1]
    dim_a2 = A.shape[2]
    dim_b1 = B.shape[1]
    dim_b2 = B.shape[2]

    A = tf.reshape(A, [batch, -1, 1])
    B = tf.reshape(B, [batch, 1, -1])
    AB = tf.matmul(A, B)
    AB = tf.reshape(AB, [batch, dim_a1, dim_a2, dim_b1, dim_b2])
    AB = tf.transpose(AB, [0, 1, 3, 2, 4])
    AB = tf.reshape(AB, [batch, dim_a1 * dim_b1, dim_a2 * dim_b2])
    
    return AB


def index_generator(n, N=None, trace=True, grid=True):
    index_list1 = np.arange(0, 6**n)
    if trace:
        index_list2 = np.arange(0, 4**n - 1)
    else:
        index_list2 = np.arange(0, 3**n)

    if grid:
        if N is None:
            N = len(index_list1) * len(index_list2)

        index_list1, index_list2 = np.meshgrid(index_list1, index_list2)
        index_list = np.vstack([index_list1.flatten(), index_list2.flatten()]).T
        np.random.shuffle(index_list)
    else:
        index_list1 = np.random.choice(index_list1, N, replace=True)
        index_list2 = np.random.choice(index_list2, N, replace=True)
        index_list = np.vstack([index_list1, index_list2]).T

    return index_list[:N, 0], index_list[:N, 1]


def train_val_split(inputs, targets, ratio=0.9):
    N = targets.shape[0]
    N_train = int(ratio * N)
    inputs_train, inputs_val = [inputs[0][:N_train], inputs[1][:N_train]], [
        inputs[0][N_train:],
        inputs[1][N_train:],
    ]
    targets_train, targets_val = targets[:N_train], targets[N_train:]

    return inputs_train, targets_train, inputs_val, targets_val


def average_tensors(tensor_list):
    pass


def saver(object, filename):
    pickle.dump(object, open(filename, "wb"))


def loader(filename):
    object = pickle.load(open(filename, "rb"))

    return object


DATA_ID = "../../data"
FIGURE_ID = "../../latex/figures"


def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)


def data_path(data_id):
    return os.path.join(DATA_ID, data_id)
