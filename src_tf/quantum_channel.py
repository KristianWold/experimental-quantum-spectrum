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
        K = kraus[:, i]
        channel += tf_kron(K, tf.math.conj(K))
    
    channel = channel[0]

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


def channel_to_choi_map(channel_list, invert_list=None):
    if not isinstance(channel_list, list):
        channel_list = [channel_list]

    if invert_list is None:
        invert_list = [False] * len(channel_list)

    d = channel_list[0].d
    super_operator_full = tf.eye(d**2, dtype=precision)

    for channel, invert in zip(channel_list, invert_list):
        super_operator = reshuffle(channel.choi)
        if invert:
            super_operator = tf.linalg.inv(super_operator)
        super_operator_full = super_operator @ super_operator_full

    choi_map = ChoiMapStatic(super_operator_full)

    return choi_map


def channel_fidelity(channel_A, channel_B):
    choi_A = channel_A.choi
    choi_B = channel_B.choi
    d_squared = choi_A.shape[0]
    fidelity = state_fidelity(choi_B, choi_A) / d_squared

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

    def spectrum(self, **kwargs):
        return channel_spectrum(self, **kwargs)

    @property
    def choi(self):
        pass
    
    @property
    def num_parameters(self):
        num_param = 0
        for param in self.parameter_list:
            dims = 1
            for dim in param.shape:
                dims *= dim
            num_param += dims
        
        return num_param
        


class ChoiMap(Channel):
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
            spam = IdealSPAM(self.d)
        self.spam = spam
        self.I = tf.eye(d, dtype=precision)

        _, self.A, self.B = generate_ginibre(d**2, rank, trainable=trainable)
        self.parameter_list = [self.A, self.B]

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.complex(self.A, self.B) / self.d

        XX = tf.matmul(G, G, adjoint_b=True)

        Y = partial_trace(XX)
        Y = tf.linalg.sqrtm(Y)
        Y = tf.linalg.inv(Y)
        Ykron = kron(self.I, Y)

        choi = Ykron @ XX @ Ykron
        self.super_operator = reshuffle(choi)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, self.d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, self.d, self.d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)


class IntegrableChoiMap(Channel):
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
            spam = IdealSPAM(self.d)
        self.spam = spam
        self.I = tf.eye(d, dtype=precision)

        _, self.A, self.B = generate_ginibre(d**2, d**2, trainable=trainable)
        self.eig = tf.Variable(tf.random.normal((d**2,), 0, 1), trainable=trainable)
        self.parameter_list = [self.A, self.B, self.eig]

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.complex(self.A, self.B) / self.d
        U = generate_unitary(G=G)
        eig = tf.cast(tf.abs(self.eig), precision)
        XX = tf.linalg.diag(eig)
        XX = tf.matmul(U, XX)
        XX = tf.matmul(XX, U, adjoint_b=True)

        Y = partial_trace(XX)
        Y = tf.linalg.sqrtm(Y)
        Y = tf.linalg.inv(Y)
        Ykron = kron(self.I, Y)

        choi = Ykron @ XX @ Ykron
        self.super_operator = reshuffle(choi)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, self.d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, self.d, self.d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)


class ChoiMapStatic(Channel):
    def __init__(
        self,
        X=None,
        mode="super_operator",
        spam=None,
    ):
        if mode == "super_operator":
            self.super_operator = X
        if mode == "choi":
            self.super_operator = reshuffle(X)
        if mode == "unitary":
            X = tf.cast(X, precision)
            self.super_operator = kron(X, tf.math.conj(X))

        self.d = int(np.sqrt(self.super_operator.shape[0]))

        if spam is None:
            spam = IdealSPAM(self.d)
        self.spam = spam

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, self.d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, self.d, self.d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)


class ReplacementChannel(Channel):
    def __init__(self, d=None, spam=None):
        self.d = d
        if spam is None:
            spam = IdealSPAM(self.d)
        self.spam = spam

        choi = tf.eye(d**2, dtype=precision) / self.d
        self.super_operator = reshuffle(choi)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, self.d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, self.d, self.d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)


class IdentityChannel(Channel):
    def __init__(self, d=None, spam=None):
        self.d = d
        if spam is None:
            spam = IdealSPAM(self.d)
        self.spam = spam

        # choi =
        # self.super_operator = reshuffle(choi)

    def apply_channel(self, state):

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)


class PTPMap:
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
        G = tf.complex(self.A, self.B) / self.d
        GG = tf.matmul(G, G, adjoint_b=True)
        GG = self.d * GG / tf.linalg.trace(GG)
        self.super_operator = reshuffle(GG)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, self.d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, self.d, self.d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)


class PMap:
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
        G = tf.complex(self.A, self.B) / self.d
        GG = tf.matmul(G, G, adjoint_b=True)
        self.super_operator = reshuffle(GG)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, self.d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, self.d, self.d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)
