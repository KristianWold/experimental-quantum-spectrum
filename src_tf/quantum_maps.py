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

def maps_to_choi(map_list):
    d = map_list[0].d
    choi = tf.zeros((d**2, d**2), dtype=tf.complex64)
    for i in range(d):
        for j in range(d):
            M = tf.zeros((d, d), dtype=tf.complex64)
            M[i,j] = 1
            M_prime = tf.copy(M)
            for map in map_list:
                M_prime = map.apply_map(M_prime)

            choi += np.kron(M_prime, M)

    return choi


def reshuffle_choi(choi):
    d = int(np.sqrt(choi.shape[0]))
    choi = tf.reshape(choi, (d,d,d,d))
    choi = tf.transpose(choi, perm = [0,2,1,3])
    choi = tf.reshape(choi, (d**2,d**2))

    return choi


def choi_spectrum(choi):
    choi = reshuffle_choi(choi)
    eig, _ = np.linalg.eig(choi)

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
                 trainable = True,
                 ):

        self.U = U
        self.d = d
        self.rank = rank

        _, self.A, self.B = generate_ginibre(rank*d, d, trainable = trainable)
        self.parameter_list = [self.A, self.B]

        if self.U is not None:
            k = np.log(1/c - 1)
            self.k = -tf.Variable(tf.cast(k, dtype = tf.double), trainable = True)
            self.parameter_list.append(self.k)
        else:
            self.k = None

        self.kraus_list = None
        self.generate_map()

    def generate_map(self):
        d = self.d
        G = self.A + 1j*self.B
        U = generate_unitary(G)

        self.kraus_list = []
        if self.U is not None:
            c = 1/(1 + tf.exp(-self.k))
            self.kraus_list.append(np.sqrt(c)*self.U)
            self.kraus_list.extend([tf.sqrt(1-c)*U[i*d:(i+1)*d, :d] for i in range(self.rank)])
        else:
            self.kraus_list.extend([U[i*d:(i+1)*d, :d] for i in range(self.rank)])

    def apply_map(self, state):

        state = sum([K@state@tf.linalg.adjoint(K) for K in self.kraus_list])
        return state
