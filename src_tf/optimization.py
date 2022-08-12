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
from loss_functions import *
from quantum_maps import *
from utils import *
from experiments import *
from set_precision import *


class ModelQuantumMap:

    def __init__(self,
                 q_map = None,
                 loss = None,
                 optimizer = None,
                 ):
        self.q_map = q_map
        self.loss = loss
        self.optimizer = optimizer
        self.loss_list = []
        self.c_list = []


#   @profile
    def train(self,
              inputs = None,
              targets = None,
              inputs_val = None,
              targets_val = None,
              num_iter = 1000,
              N = 1,
              verbose = True):

        indices = tf.range(targets.shape[0])

        for step in tqdm(range(num_iter)):


            batch = tf.random.shuffle(indices)[:N]
            inputs_batch = [tf.gather(data, batch, axis=0) for data in inputs]
            targets_batch = tf.gather(targets, batch, axis=0)

            with tf.GradientTape() as tape:
                self.q_map.generate_map()
                loss = self.loss(self.q_map, inputs_batch, targets_batch)

            grads = tape.gradient(loss, self.q_map.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.q_map.parameter_list))

            if targets_val is None:
                pass
            elif len(targets_val) == 1:
                loss = channel_fidelity(self.q_map, targets_val[0])


            self.loss_list.append(np.abs(loss.numpy()))
            if self.q_map.c is not None:
                self.c_list.append(np.abs(self.q_map.c.numpy()))
            if inputs_val is None:
                loss_val = 0
            else:
                loss_val = np.abs(self.loss(self.q_map, inputs_val, targets_val).numpy())
            if verbose:
                print(f"Step:{step}, train: {np.abs(loss.numpy()):.5f}, val: {loss_val:.5f}")

        self.q_map.generate_map()


class POVM:
    def __init__(self, d, optimizer, trainable=True):
        self.d = d
        self.A = tf.cast(tf.random.normal((d, d, d), 0, 1), dtype = precision)
        self.B = tf.cast(tf.random.normal((d, d, d), 0, 1), dtype = precision)

        if trainable:
            self.A = tf.Variable(self.A, trainable = True)
            self.B = tf.Variable(self.B, trainable = True)

        self.parameter_list = [self.A, self.B]
        self.optimizer = optimizer

        self.generate_POVM()

    def generate_POVM(self):
        G = self.A + 1j*self.B
        AA = tf.matmul(G, G, adjoint_b=True)
        D = tf.math.reduce_sum(AA, axis = 0)
        invsqrtD = tf.linalg.inv(tf.linalg.sqrtm(D))
        self.povm = tf.matmul(tf.matmul(invsqrtD, AA), invsqrtD)

    def train(self, num_iter, inputs, targets, N = 1):
        indices = tf.range(targets.shape[0])

        for step in tqdm(range(num_iter)):
            batch = tf.random.shuffle(indices)[:N]
            inputs_batch = tf.gather(inputs, batch, axis=0)
            targets_batch = tf.gather(targets, batch, axis=0)

            with tf.GradientTape() as tape:
                self.generate_POVM()
                outputs = measurement(inputs_batch, povm=self.povm)

                loss = self.d**2*tf.math.reduce_mean((outputs - targets_batch)**2)

            grads = tape.gradient(loss, self.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.parameter_list))
            print(step, np.abs(loss.numpy()))

        self.generate_POVM()


class SPAM:
    def __init__(self,
                 d=None,
                 init = None,
                 povm = None,
                 use_corr_mat = False,
                 optimizer = None,
                 trainable = True):

        self.d = d
        self.use_corr_mat = use_corr_mat

        self.parameter_list = []
        if init is None:
            self.A = tf.cast(tf.random.normal((d, d), 0, 1), dtype = precision)
            self.B = tf.cast(tf.random.normal((d, d), 0, 1), dtype = precision)

            self.A = tf.Variable(self.A, trainable = True)
            self.B = tf.Variable(self.B, trainable = True)

            self.parameter_list.extend([self.A, self.B])
        else:
            self.A = self.B = None
            self.init = init


        if povm is None:
            if not use_corr_mat:
                self.C = tf.cast(tf.random.normal((d, d, d), 0, 1), dtype = precision)
                self.C = tf.Variable(self.C, trainable = True)

                self.D = tf.cast(tf.random.normal((d, d, d), 0, 1), dtype = precision)
                self.D = tf.Variable(self.D, trainable = True)

                self.parameter_list.extend([self.C, self.D])
            else:
                self.C = tf.cast(tf.random.normal((d, d), 0, 1), dtype = precision)
                self.C = tf.Variable(self.C, trainable = True)
                self.parameter_list.extend([self.C])


        else:
            self.C = self.D = None
            self.povm = povm

        self.optimizer = optimizer

        self.generate_SPAM()

    def generate_SPAM(self):

        if self.A is not None:
            X = self.A + 1j*self.B
            XX = tf.matmul(X, X, adjoint_b=True)
            state = XX/tf.linalg.trace(XX)
            self.init = state

        if self.C is not None:
            if not self.use_corr_mat:
                X = self.C + 1j*self.D
                XX = tf.matmul(X, X, adjoint_b=True)
                D = tf.math.reduce_sum(XX, axis = 0)
                invsqrtD = tf.linalg.inv(tf.linalg.sqrtm(D))
                self.povm = tf.matmul(tf.matmul(invsqrtD, XX), invsqrtD)

            else:
                X = tf.abs(self.C)
                X = X/tf.reduce_sum(X, axis = 1)
                corr_mat = tf.transpose(X)
                self.povm = corr_mat_to_povm(corr_mat)

    def train(self, num_iter, inputs, targets, N = 1):
        indices = tf.range(targets.shape[0])

        for step in tqdm(range(num_iter)):
            batch = tf.random.shuffle(indices)[:N]
            inputs_batch = tf.gather(inputs, batch, axis=0)
            targets_batch = tf.gather(targets, batch, axis=0)

            with tf.GradientTape() as tape:
                self.generate_SPAM()
                outputs = measurement(tf.repeat(self.init[None,:,:], N, axis=0),
                                      U_basis = inputs_batch,
                                      povm = self.povm)

                loss = self.d**2*tf.math.reduce_mean((outputs - targets_batch)**2)

            grads = tape.gradient(loss, self.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.parameter_list))
            print(step, np.abs(loss.numpy()))

        self.generate_SPAM()

    def pretrain(self, num_iter, targets):
        init_target, povm_target = targets
        for step in tqdm(range(num_iter)):

            with tf.GradientTape() as tape:
                self.generate_SPAM()
                loss1 = tf.reduce_mean(tf.abs(self.init - init_target)**2)
                loss2 = tf.reduce_mean(tf.abs(self.povm - povm_target)**2)
                loss = loss1 + loss2

            grads = tape.gradient(loss, self.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.parameter_list))
            print(step, np.abs(loss.numpy()))

        self.generate_SPAM()
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))
