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
                 povm = None,
                 trainable = True,
                 ):

        self.U = U
        self.d = d
        self.rank = rank
        
        if povm is None:
            self.povm = povm_ideal(int(np.log2(d)))
        else:
            self.povm = povm

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
        U = generate_unitary(G)
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
