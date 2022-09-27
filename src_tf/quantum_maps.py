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
    rank = kraus.shape[1]
    choi = 0
    print(kraus.shape)

    for i in range(rank):
        K = kraus[0, i]
        choi = choi + tf.experimental.numpy.kron(K, tf.linalg.adjoint(K))

    if reshuffle:
        choi = reshuffle_choi(choi)

    return choi


def maps_to_choi(map_list):
    d = map_list[0].d
    choi = tf.zeros((d**2, d**2), dtype=precision)
    M = np.zeros((d**2, d, d))
    for i in range(d):
        for j in range(d):

            M[d*i + j, i, j] = 1

    M = tf.convert_to_tensor(M, dtype=precision)
    M_prime = tf.identity(M)
    for map in map_list:
        M_prime = map.apply_map(M_prime)
    for i in range(d**2):
        choi += tf.experimental.numpy.kron(M_prime[i], M[i])

    return choi



def choi_spectrum(choi, resuffle=True):
    choi = reshuffle_choi(choi)
    eig, _ = tf.linalg.eig(choi)

    x = tf.cast(tf.math.real(eig), dtype=precision)
    y = tf.cast(tf.math.imag(eig), dtype=precision)

    return [x, y]


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


class RegularizedKrausMap():

    def __init__(self,
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
        self.C = tf.cast(tf.random.normal((1, rank), 0, 1), dtype = precision)
        self.C = tf.Variable(self.reg, trainable=True)
        self.parameter_list = [self.A, self.B, self.C]

        self.kraus = None
        self.generate_map()


    def generate_map(self):
        d = self.d
        G = self.A + 1j*self.B
        U = generate_unitary(G=G)
        self.kraus = tf.reshape(U, (1, self.rank, self.d, self.d))
        
        reg = tf.abs(self.C)
        reg = self.d*reg/tf.reduce_sum(reg)

        KK = tf.matmul(self.kraus, self.kraus, adjoint_a=True)
        TrKK = tf.linalg.trace(KK)

        self.kraus = reg*self.kraus/KK


    def apply_map(self, state):
        state = tf.expand_dims(state, axis=1)
        Kstate = tf.matmul(self.kraus, state)
        KstateK = tf.matmul(Kstate, self.kraus, adjoint_b=True)
        state = tf.reduce_sum(KstateK, axis=1)

        return state



