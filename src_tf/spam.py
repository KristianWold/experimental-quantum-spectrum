import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import tensorflow as tf
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm
from copy import deepcopy
from set_precision import *


from quantum_tools import *
from utils import *
from set_precision import *


def generate_corruption_matrix(counts_list):
    n = len(list(counts_list[0].keys())[0])
    corr_mat = np.zeros((2**n, 2**n))
    for i, counts in enumerate(counts_list):
        for string, value in counts.items():
            index = int(string, 2)
            corr_mat[index, i] = value

    corr_mat = corr_mat / sum(counts_list[0].values())
    return corr_mat


def corr_mat_to_povm(corr_mat):
    d = corr_mat.shape[0]
    povm = []
    for i in range(d):
        M = tf.linalg.diag(corr_mat[i, :])
        povm.append(M)

    povm = tf.convert_to_tensor(povm, dtype=precision)

    return povm


def init_ideal(d):
    init = np.zeros((d, d))
    init[0, 0] = 1
    init = tf.convert_to_tensor(init, dtype=precision)
    return init


def povm_ideal(d):
    povm = tf.cast(corr_mat_to_povm(np.eye(d)), dtype=precision)
    return povm


class InitialState:
    def __init__(self, d, c = 0.9, trainable=True):
        self.d = d

        self.A = tf.cast(tf.random.normal((d, d), 0, 1), dtype=precision)
        self.B = tf.cast(tf.random.normal((d, d), 0, 1), dtype=precision)
        self.init_ideal = init_ideal(d)

        if trainable:
            self.A = tf.Variable(self.A, trainable=True)
            self.B = tf.Variable(self.B, trainable=True)

        self.parameter_list = [self.A, self.B]

        k = -np.log(1 / c - 1)
        self.k = tf.Variable(k, trainable=True)
        self.parameter_list.append(self.k)

        self.generate_init()

    def generate_init(self):
        G = self.A + 1j * self.B
        AA = tf.matmul(G, G, adjoint_b=True)
        self.init = AA/tf.linalg.trace(AA)
        self.init = self.c * self.init + (1 - self.c) * self.init_ideal

    @property
    def c(self):
        return tf.cast(1 / (1 + tf.exp(-self.k)),dtype=precision)


class POVM:
    def __init__(self, d, c = 0.9, trainable=True):
        self.d = d
        self.A = tf.cast(tf.random.normal((d, d, d), 0, 1), dtype=precision)
        self.B = tf.cast(tf.random.normal((d, d, d), 0, 1), dtype=precision)
        self.povm_ideal = povm_ideal(d)

        if trainable:
            self.A = tf.Variable(self.A, trainable=True)
            self.B = tf.Variable(self.B, trainable=True)

        self.parameter_list = [self.A, self.B]

        k = -np.log(1 / c - 1)
        self.k = tf.Variable(k, trainable=True)
        self.parameter_list.append(self.k)

        self.generate_POVM()

    def generate_POVM(self):
        G = self.A + 1j * self.B
        AA = tf.matmul(G, G, adjoint_b=True)
        D = tf.math.reduce_sum(AA, axis=0)
        invsqrtD = tf.linalg.inv(tf.linalg.sqrtm(D))
        self.povm = tf.matmul(tf.matmul(invsqrtD, AA), invsqrtD)
        self.povm = self.c*self.povm_ideal + (1-self.c)*self.povm

    @property
    def c(self):
        return tf.cast(1 / (1 + tf.exp(-self.k)),dtype=precision)


class SPAM:
    def __init__(
        self, init = None, povm = None, loss_function = None, optimizer=None
    ):

        self.d = init.d
        self.init = init
        self.povm = povm

        self.loss_function = loss_function
        self.optimizer = optimizer
        self.parameter_list = self.init.parameter_list + self.povm.parameter_list

        self.generate_SPAM()

    def train(self, num_iter, inputs, targets, N=None, verbose=True):
        if N is None:
            N = targets.shape[0]
        indices = tf.range(targets.shape[0])

        for step in tqdm(range(num_iter)):
            batch = tf.random.shuffle(indices)[:N]
            inputs_batch = tf.gather(inputs, batch, axis=0)
            targets_batch = tf.gather(targets, batch, axis=0)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.parameter_list)
                self.generate_SPAM()
                inputs_batch = apply_unitary(self.init.init, inputs_batch)
                output_batch = measurement(inputs_batch, povm = self.povm.povm)
                loss = tf.math.reduce_mean((targets_batch - output_batch)**2)

            grads = tape.gradient(loss, self.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.parameter_list))
            if verbose:
                print(step, np.abs(loss.numpy()))
        print(np.abs(loss.numpy()))

        self.generate_SPAM()

    def generate_SPAM(self):
        self.init.generate_init()
        self.povm.generate_POVM()

