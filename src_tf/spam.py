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
    povm = corr_mat_to_povm(np.eye(d))
    return povm


class POVM:
    def __init__(self, d, optimizer, trainable=True):
        self.d = d
        self.A = tf.cast(tf.random.normal((d, d, d), 0, 1), dtype=precision)
        self.B = tf.cast(tf.random.normal((d, d, d), 0, 1), dtype=precision)

        if trainable:
            self.A = tf.Variable(self.A, trainable=True)
            self.B = tf.Variable(self.B, trainable=True)

        self.parameter_list = [self.A, self.B]
        self.optimizer = optimizer

        self.generate_POVM()

    def generate_POVM(self):
        G = self.A + 1j * self.B
        AA = tf.matmul(G, G, adjoint_b=True)
        D = tf.math.reduce_sum(AA, axis=0)
        invsqrtD = tf.linalg.inv(tf.linalg.sqrtm(D))
        self.povm = tf.matmul(tf.matmul(invsqrtD, AA), invsqrtD)

    def train(self, num_iter, inputs, targets, N=1):
        indices = tf.range(targets.shape[0])

        for step in tqdm(range(num_iter)):
            batch = tf.random.shuffle(indices)[:N]
            inputs_batch = tf.gather(inputs, batch, axis=0)
            targets_batch = tf.gather(targets, batch, axis=0)

            with tf.GradientTape() as tape:
                self.generate_POVM()
                outputs = measurement(inputs_batch, povm=self.povm)

                loss = self.d**2 * tf.math.reduce_mean((outputs - targets_batch) ** 2)

            grads = tape.gradient(loss, self.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.parameter_list))
            print(step, np.abs(loss.numpy()))

        self.generate_POVM()


class SPAM:
    def __init__(
        self, d=None, init="random", povm="random", use_corr_mat=False, optimizer=None
    ):

        self.d = d
        self.use_corr_mat = use_corr_mat
        self.n = int(np.log2(d))

        self.parameter_list = []
        if init is "random":
            self.A = tf.Variable(
                tf.cast(tf.random.normal((d, d), 0, 1), dtype=precision)
            )
            self.B = tf.Variable(
                tf.cast(tf.random.normal((d, d), 0, 1), dtype=precision)
            )
            self.parameter_list.extend([self.A, self.B])
        elif init is "ideal":
            self.A = np.zeros((d, d))
            self.A[0, 0] = 1
            self.A = tf.Variable(tf.cast(self.A, dtype=precision))
            self.B = tf.Variable(tf.zeros_like(self.A, dtype=precision))
            self.parameter_list.extend([self.A, self.B])
        else:
            self.A = self.B = None
            self.init = init

        if povm is "random":
            if not use_corr_mat:
                self.C = tf.Variable(
                    tf.cast(tf.random.normal((d, d, d), 0, 1), dtype=precision)
                )
                self.D = tf.Variable(
                    tf.cast(tf.random.normal((d, d, d), 0, 1), dtype=precision)
                )
                self.parameter_list.extend([self.C, self.D])
            else:
                self.C = tf.Variable(
                    tf.cast(tf.random.normal((d, d), 0, 1), dtype=precision)
                )
                self.parameter_list.extend([self.C])
        elif povm is "ideal":
            if not use_corr_mat:
                self.C = np.zeros((d, d, d))
                for i in range(d):
                    self.C[i, i, i] = 1
                self.C = tf.Variable(tf.cast(self.C, dtype=precision))
                self.D = tf.Variable(tf.zeros_like(self.C, dtype=precision))
                self.parameter_list.extend([self.C, self.D])
            else:
                self.C = tf.Variable(tf.cast(tf.eye(d), dtype=precision))
                self.parameter_list.extend([self.C])

        else:
            self.C = self.D = None
            self.povm = povm

        self.optimizer = optimizer
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
                outputs = measurement(
                    tf.repeat(self.init[None, :, :], N, axis=0),
                    U_basis=inputs_batch,
                    povm=self.povm,
                )

                loss = self.d**2 * tf.math.reduce_mean((outputs - targets_batch) ** 2)

            grads = tape.gradient(loss, self.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.parameter_list))
            if verbose:
                print(step, np.abs(loss.numpy()))
        print(np.abs(loss.numpy()))

        self.generate_SPAM()

    def pretrain(self, num_iter, targets, verbose=True):
        init_target, povm_target = targets
        for step in range(num_iter):

            with tf.GradientTape() as tape:
                self.generate_SPAM()
                loss1 = tf.reduce_mean(tf.abs(self.init - init_target) ** 2)
                loss2 = tf.reduce_mean(tf.abs(self.povm - povm_target) ** 2)
                loss = loss1 + loss2

            grads = tape.gradient(loss, self.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.parameter_list))

        self.generate_SPAM()
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))

    def generate_SPAM(self):
        if self.A is not None:
            self.generate_init()

        if self.C is not None:
            self.generate_POVM()

    def generate_init(self):
        X = self.A + 1j * self.B
        XX = tf.matmul(X, X, adjoint_b=True)
        state = XX / tf.linalg.trace(XX)
        self.init = state

    def generate_POVM(self):
        if not self.use_corr_mat:
            X = self.C + 1j * self.D
            XX = tf.matmul(X, X, adjoint_b=True)
            D = tf.math.reduce_sum(XX, axis=0)
            invsqrtD = tf.linalg.inv(tf.linalg.sqrtm(D))
            self.povm = tf.matmul(tf.matmul(invsqrtD, XX), invsqrtD)

        else:
            X = tf.abs(self.C)
            X = X / tf.reduce_sum(X, axis=1)
            corr_mat = tf.transpose(X)
            self.povm = corr_mat_to_povm(corr_mat)
