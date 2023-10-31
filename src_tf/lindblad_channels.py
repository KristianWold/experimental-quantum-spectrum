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
from quantum_channel import *
from spam import *
from utils import *
from set_precision import *


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
            spam = IdealSPAM(d)
        self.spam = spam

        _, self.A, self.B = generate_ginibre(d, d, trainable=trainable)
        _, self.C, self.D = generate_ginibre(d**2, rank, trainable=trainable)

        self.parameter_list = [self.A, self.B, self.C, self.D]

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
            -1j * self.alpha * (tf_kron(self.I, H)[0] - tf_kron(tf.transpose(H), self.I)[0])
            + phi
            - 0.5 * (tf_kron(tf.transpose(phi_star), self.I)[0] + tf_kron(self.I, phi_star)[0])
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


class TracelessLindbladMap(Channel):
    def __init__(
        self,
        d=None,
        rank=None,
        weight=None,
        spam=None,
        trainable=True,
        jump_operators_trainable=True,
        generate=True,
    ):

        self.d = d
        self.rank = rank
        self.weight = weight
        self.I = tf.cast(tf.eye(d), dtype=precision)

        if spam is None:
            spam = IdealSPAM(d)
        self.spam = spam

        _, A, B = generate_ginibre(d, d, trainable=trainable)
        self.H_params = [tf.cast(A, dtype=tf.double), tf.cast(B, dtype=tf.double)]

        self.A_list = []
        self.B_list = []
        for i in range(rank - 1):
            _, A, B = generate_ginibre(d, d, trainable=jump_operators_trainable)
            self.A_list.append(tf.cast(A, dtype=tf.double))
            self.B_list.append(tf.cast(B, dtype=tf.double))

        self.parameter_list = self.H_params

        if jump_operators_trainable:
            self.parameter_list += self.A_list + self.B_list

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        H = self.generate_hamiltonian()
        L_list = self.generate_jump_operators()

        HH = -1j * (kron(H, self.I) - kron(self.I, tf.math.conj(H)))
        LL = 0
        for i, L in enumerate(L_list):
            L2 = tf.matmul(L, L, adjoint_a=True)
            LL += kron(L, tf.math.conj(L)) - 0.5 * (
                (kron(L2, self.I) + kron(self.I, tf.math.conj(L2)))
            )

        self.super_operator = tf.linalg.expm(HH + self.weight * LL)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, d, d))

        return state

    def generate_hamiltonian(self):
        G = tf.complex(self.H_params[0], self.H_params[1])
        H = G + tf.linalg.adjoint(G)
        return H

    def generate_jump_operators(self):
        d = self.A_list[0].shape[0]
        L_list = [tf.complex(A, B) for A, B in zip(self.A_list, self.B_list)]

        L_list = [L - tf.linalg.trace(L) * tf.eye(d) for L in L_list]

        return L_list

    @property
    def choi(self):
        return reshuffle(self.super_operator)
