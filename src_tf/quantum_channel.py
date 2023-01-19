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
from spam import *
from utils import *
from set_precision import *


def reshuffle(A):
    d = int(np.sqrt(A.shape[0]))
    A = tf.reshape(A, (d, d, d, d))
    A = tf.einsum("jklm -> jlkm", A)
    A = tf.reshape(A, (d**2, d**2))

    return A


def kraus_to_choi(kraus_channel, use_reshuffle=True):
    kraus = kraus_channel.kraus
    rank = kraus.shape[1]
    channel = 0

    for i in range(rank):
        K = kraus[0, i]
        channel += tf.experimental.numpy.kron(K, tf.math.conj(K))

    if use_reshuffle:
        choi = reshuffle(channel)

    return choi


def state_purity(A):
    eig, _ = tf.linalg.eig(A)
    purity = tf.math.reduce_sum(eig**2)
    return purity


def effective_rank(channel):
    choi = channel.choi
    d2 = choi.shape[0]

    purity = state_purity(choi)

    rank_eff = d2 / purity
    return rank_eff

"""
def channel_to_choi(channel_list):
    if not isinstance(channel, list):
        channel_list = [channel_list]

    d = channel_list[0].d
    choi = tf.zeros((d**2, d**2), dtype=precision)
    M = np.zeros((d**2, d, d))
    for i in range(d):
        for j in range(d):

            M[d * i + j, i, j] = 1

    M = tf.convert_to_tensor(M, dtype=precision)
    M_prime = tf.identity(M)
    for channel in channel_list:
        M_prime = channel.apply_channel(M_prime)
    for i in range(d**2):
        choi += tf.experimental.numpy.kron(M_prime[i], M[i])

    return choi
"""


def channel_to_choi_map(channel_list):
    if not isinstance(channel_list, list):
        channel_list = [channel_list]

    d = channel_list[0].d
    super_operator_full = reshuffle(channel_list[0].choi)
    for channel in channel_list[1:]:
        super_operator = reshuffle(channel.choi)
        super_operator_full = super_operator_full@super_operator
    
    choi_map = ChoiMapStatic(super_operator_full)

    return choi_map


def channel_fidelity(channel_A, channel_B):
    choi_A = channel_A.choi
    choi_B = channel_B.choi
    d_squared = choi_A.shape[0]
    fidelity = state_fidelity(choi_A, choi_B) / d_squared

    return fidelity


def channel_steady_state(channel):
    d = channel.d
    super_operator = reshuffle(channel.choi)
    eig, eig_vec = np.linalg.eig(super_operator)
    steady_index = tf.math.argmax(tf.abs(eig))

    steady_state = eig_vec[:, steady_index]
    steady_state = steady_state.reshape(d, d)
    steady_state = steady_state / tf.linalg.trace(steady_state)

    return steady_state


def dilute_channel(U, c, kraus_map):
    pass


class Channel:
    def __init__(self, d, rank, spam):
        self.d = d
        self.rank = rank
        self.spam = spam

        self.channel = None
        self.generate_channel()

    def generate_channel(self):
        pass

    def apply_channel(self, state):
        pass

    @property
    def choi(self):
        pass


class ChoiMap:
    def __init__(
        self,
        d=None,
        rank=None,
        spam=None,
        trainable=True,
        generate=True,
    ):

        self.d = d
        self.rank = rank

        if spam is None:
            spam = SPAM(d=d, init=init_ideal(d), povm=povm_ideal(d))
        self.spam = spam

        _, self.A, self.B = generate_ginibre(d**2, rank, trainable=trainable)
        self.parameter_list = [self.A, self.B]

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = self.A + 1j * self.B

        XX = tf.matmul(G, G, adjoint_b=True)

        Y = partial_trace(XX)
        Y = tf.linalg.sqrtm(Y)
        Y = tf.linalg.inv(Y)
        Ykron = kron(self.I, Y)

        choi = Ykron @ XX @ Ykron
        self.super_operator = reshuffle(choi)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, d, d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)


class ChoiMapStatic(Channel):
    def __init__(
        self,
        super_operator=None,
    ):
        self.super_operator = super_operator
        self.d = int(np.sqrt(super_operator.shape[0]))

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, d, d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)