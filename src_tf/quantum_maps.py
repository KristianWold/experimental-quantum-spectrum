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

from quantum_tools import *
from loss_functions import *
from utils import *
from experiments import *
from set_precision import *


def reshuffle_choi(choi):
    d = int(np.sqrt(choi.shape[0]))
    choi = tf.reshape(choi, (d,d,d,d))
    choi = tf.transpose(choi, perm = [0,2,1,3])
    choi = tf.reshape(choi, (d**2,d**2))

    return choi


def kraus_to_choi(kraus_map, reshuffle = True):
    kraus = kraus_map.kraus
    rank = kraus.shape[0]
    choi = 0

    for i in range(rank):
        K = kraus[0, i]
        choi = choi + tf.experimental.numpy.kron(tf.linalg.adjoint(K), K)

    if reshuffle:
        choi = reshuffle_choi(choi)

    return choi


def choi_spectrum(choi):
    choi = reshuffle_choi(choi)
    eig, _ = tf.linalg.eig(choi)

    x = np.real(eig)
    y = np.imag(eig)

    return np.array([x, y])


def choi_steady_state(choi):
    d = int(np.sqrt(choi.shape[0]))
    choi = reshuffle_choi(choi)
    _, eig_vec = np.linalg.eig(choi)

    steady_state = eig_vec[:,0]
    steady_state = steady_state.reshape(d, d)

    return steady_state


class KrausMap():

    def __init__(self,
                 U = None,
                 c = None,
                 d = None,
                 rank = None,
                 spam = None,
                 trainable = True,
                 ):

        self.U = U
        self.d = d
        self.rank = rank

        if spam is None:
            spam = SPAM(d=d,
                        init = init_ideal(d),
                        povm = povm_ideal(d))
        self.spam = spam

        _, self.A, self.B = generate_ginibre(rank*d, d, trainable = trainable)
        self.parameter_list = [self.A, self.B]

        if self.U is not None:
            self.U = tf.expand_dims(tf.expand_dims(self.U, 0), 0)
            k = -np.log(1/c - 1)
            self.k = tf.Variable(tf.cast(k, dtype = precision), trainable = True)
            self.parameter_list.append(self.k)
        else:
            self.k = None

        self.kraus = None
        self.generate_map()

    def generate_map(self):
        d = self.d
        G = self.A + 1j*self.B
        U = generate_unitary(G=G)
        self.kraus = tf.reshape(U, (1, self.rank, self.d, self.d))

        if self.U is not None:
            c = 1/(1 + tf.exp(-self.k))
            self.kraus = tf.concat([tf.sqrt(c)*self.U, tf.sqrt(1-c)*self.kraus], axis=1)

    @property
    def c(self):
        if self.k is None:
            c = None
        else:
            c = 1/(1 + tf.exp(-self.k))
        return c


    def apply_map(self, state):
        state = tf.expand_dims(state, axis=1)
        Kstate = tf.matmul(self.kraus, state)
        KstateK = tf.matmul(Kstate, self.kraus, adjoint_b=True)
        state = tf.reduce_sum(KstateK, axis=1)

        return state


class SPAM:
    def __init__(self,
                 d=None,
                 init = None,
                 povm = None,
                 use_corr_mat = False,
                 optimizer = None):

        self.d = d
        self.use_corr_mat = use_corr_mat

        self.parameter_list = []
        if init is None:
            self.A =  tf.Variable(tf.cast(tf.random.normal((d, d), 0, 1), dtype = precision))
            self.B =  tf.Variable(tf.cast(tf.random.normal((d, d), 0, 1), dtype = precision))
            self.parameter_list.extend([self.A, self.B])
        else:
            self.A = self.B = None
            self.init = init

        if povm is None:
            if not use_corr_mat:
                self.C = tf.Variable(tf.cast(tf.random.normal((d, d, d), 0, 1), dtype = precision))
                self.D = tf.Variable(tf.cast(tf.random.normal((d, d, d), 0, 1), dtype = precision))
                self.parameter_list.extend([self.C, self.D])
            else:
                self.C =  tf.Variable(tf.cast(tf.random.normal((d, d), 0, 1), dtype = precision))
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

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.parameter_list)
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
