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
        self.I = tf.cast(tf.eye(self.d), dtype=precision)

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


class CompactLindbladMap(Channel):
    def __init__(
        self,
        d=None,
        rank=None,
        alpha=1,
        beta=1,
        spam=None,
        trainable=True,
        generate=True,
    ):

        self.d = d
        self.rank = rank
        self.alpha = alpha
        self.beta = beta

        self.n = int(np.log2(d))

        self.I = tf.cast(tf.eye(d), dtype=precision)

        if spam is None:
            spam = SPAM(init=InitialState(d, c=0.99999), povm=POVM(d, c=0.99999))
        self.spam = spam

        _, self.A, self.B = generate_ginibre(d, d, trainable=trainable)
        _, self.C, self.D = generate_ginibre(d**2, rank - 1, trainable=trainable)
        _, self.a, _ = generate_ginibre(1, 1, trainable=trainable, complex=False)

        self.parameter_list = [self.A, self.B, self.C, self.D, self.a]

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.cast(self.A, dtype=precision) + 1j * tf.cast(self.B, dtype=precision)
        H = (G + tf.linalg.adjoint(G)) / 2

        G = tf.cast(self.C, dtype=precision) + 1j * tf.cast(self.D, dtype=precision)
        choi = tf.matmul(G, G, adjoint_b=True)
        phi = reshuffle(choi)
        phi_star = partial_trace(choi)
        expo = (
            -1j * self.alpha * (kron(self.I, H) - kron(tf.transpose(H), self.I))
            + phi
            - 0.5 * (kron(tf.transpose(phi_star), self.I) + kron(self.I, phi_star))
        )

        self.super_operator = tf.linalg.expm(self.beta * expo)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, self.d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, self.d, self.d))

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)


class ExplicitLindbladMap(Channel):
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
        self.I = tf.cast(tf.eye(d), dtype=precision)

        if spam is None:
            spam = SPAM(d=d, init=init_ideal(d), povm=povm_ideal(d))
        self.spam = spam

        _, A, B = generate_ginibre(d, d, trainable=trainable)
        self.H_params = [tf.cast(A, dtype=tf.double), tf.cast(B, dtype=tf.double)]
        _, A, _ = generate_ginibre(rank - 1, 1, trainable=trainable, complex=False)
        self.gamma_params = [tf.cast(A, dtype=tf.double)]

        self.A_list = []
        self.B_list = []
        for i in range(rank - 1):
            _, A, B = generate_ginibre(d, d, trainable=trainable)
            self.A_list.append(tf.cast(A, dtype=tf.double))
            self.B_list.append(tf.cast(B, dtype=tf.double))

        self.parameter_list = (
            self.H_params + self.gamma_params + self.A_list + self.B_list
        )

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        H = self.generate_hamiltonian()
        weights = self.generate_weights()
        L_list = self.generate_jump_operators()

        generator = -1j * (kron(H, self.I) - kron(self.I, tf.math.conj(H)))
        for i, L in enumerate(L_list):
            L2 = tf.matmul(L, L, adjoint_a=True)
            generator += weights[i] * (
                kron(L, tf.math.conj(L))
                - 0.5 * ((kron(L2, self.I) + kron(self.I, tf.math.conj(L2))))
            )

        self.super_operator = tf.linalg.expm(generator)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, d, d))

        return state

    def generate_hamiltonian(self):
        G = tf.complex(self.H_params[0], self.H_params[1])
        H = G + tf.linalg.adjoint(G)
        return H

    def generate_weights(self):
        weights = tf.cast(tf.abs(self.gamma_params[0]), dtype=precision)
        return weights

    def generate_jump_operators(self):
        L_list = [tf.complex(A, B) for A, B in zip(self.A_list, self.B_list)]

        ab = tf.linalg.trace(
            tf.matmul(self.I / np.sqrt(self.d), L_list[0], adjoint_a=True)
        )
        L_list[0] = L_list[0] - ab * self.I / np.sqrt(self.d)
        L_list[0] = L_list[0] / tf.math.sqrt(
            tf.linalg.trace(tf.matmul(L_list[0], L_list[0], adjoint_a=True))
        )

        for i in range(1, len(L_list)):
            for j in range(i):
                ab = tf.linalg.trace(tf.matmul(L_list[j], L_list[i], adjoint_a=True))
                L_list[i] += -ab * L_list[j]

            L_list[i] = L_list[i] / tf.math.sqrt(
                tf.linalg.trace(tf.matmul(L_list[i], L_list[i], adjoint_a=True))
            )

        return L_list

    @property
    def choi(self):
        return reshuffle(self.super_operator)
