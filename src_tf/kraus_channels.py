import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import tensorflow as tf
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator, random_unitary
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm

from quantum_tools import *
from spam import *
from utils import *
from set_precision import *
from quantum_channel import *


class UnitaryMap(Channel):
    def __init__(
        self,
        d=None,
        spam=None,
        trainable=True,
        generate=True,
    ):

        self.d = d

        if spam is None:
            spam = SPAM(d=d, init=init_ideal(d), povm=povm_ideal(d))
        self.spam = spam

        _, self.A, self.B = generate_ginibre(d, d, trainable=trainable)
        self.parameter_list = [self.A, self.B]

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.cast(self.A, dtype=precision) + 1j * tf.cast(self.B, dtype=precision)
        self.U = tf.reshape(generate_unitary(G=G), (1, 1, self.d, self.d))

    def apply_channel(self, state):
        U = tf.reshape(self.U, (1, self.d, self.d))
        Ustate = tf.matmul(self.U, state)
        UstateU = tf.matmul(Ustate, self.U, adjoint_b=True)
        state = tf.reduce_sum(UstateU, axis=1)

        return state

    @property
    def choi(self):
        choi = tf.experimental.numpy.kron(self.U, tf.math.conj(self.U))
        return choi


class KrausMap(Channel):
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
        self.trainable = trainable

        if spam is None:
            spam = IdealSPAM(d=d)
        self.spam = spam

        _, self.A, self.B = generate_ginibre(rank * d, d, trainable=trainable)

        self.parameter_list = []
        if self.trainable:
            self.parameter_list = [self.A, self.B]

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.complex(self.A, self.B)
        U = generate_unitary(G=G)
        self.kraus = tf.reshape(U[:, : self.d], (1, self.rank, self.d, self.d))

    def apply_channel(self, state):
        state = tf.expand_dims(state, axis=1)
        Kstate = tf.matmul(self.kraus, state)
        KstateK = tf.matmul(Kstate, self.kraus, adjoint_b=True)
        state = tf.reduce_sum(KstateK, axis=1)

        return state

    @property
    def choi(self):
        return kraus_to_choi(self)


class DilutedKrausMap(KrausMap):
    def __init__(
        self,
        U=None,
        c=None,
        kraus_part=None,
        spam=None,
        trainable=True,
        generate=True,
    ):

        self.d = U.shape[0]
        self.rank = kraus_part.rank
        self.spam = spam
        self.kraus_part = kraus_part

        if spam is None:
            spam = IdealSPAM(d=self.d)
        self.spam = spam

        self.parameter_list = self.kraus_part.parameter_list

        self.U = U
        if self.U is not None:
            self.U = tf.expand_dims(tf.expand_dims(self.U, 0), 0)
            k = -np.log(1 / c - 1)
            self.k = tf.Variable(k, trainable=True)
            self.parameter_list.append(self.k)
        else:
            self.k = None

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        if self.kraus_part.trainable:
            self.kraus_part.generate_channel()

        if self.U is not None:
            c = 1 / (1 + tf.exp(-self.k))
            c = tf.cast(c, dtype=precision)
            self.kraus = tf.concat(
                [tf.sqrt(c) * self.U, tf.sqrt(1 - c) * self.kraus_part.kraus], axis=1
            )

    @property
    def c(self):
        if self.k is None:
            c = None
        else:
            c = 1 / (1 + tf.exp(-self.k))
        return c

    @property
    def choi(self):
        return kraus_to_choi(self)


class ExtractedKrausMap(KrausMap):
    def __init__(
        self,
        d=None,
        rank=None,
        spam=None,
        trainable=True,
        generate=True,
    ):

        KrausMap.__init__(
            self, d=d, rank=rank - 1, spam=spam, trainable=trainable, generate=False
        )
        self.UnitaryMap = UnitaryMap(d=d, trainable=trainable, generate=False)
        _, self.k, _ = generate_ginibre(1, 1, trainable=trainable, complex=False)

        self.parameter_list.extend(self.UnitaryMap.parameter_list)
        self.parameter_list.append(self.k)

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        KrausMap.generate_channel(self)
        self.UnitaryMap.generate_channel()

        c = 1 / (1 + tf.exp(-self.k))
        c = tf.cast(c, dtype=precision)
        self.kraus = tf.concat(
            [tf.sqrt(c) * self.UnitaryMap.U, tf.sqrt(1 - c) * self.kraus], axis=1
        )

    @property
    def c(self):
        if self.k is None:
            c = None
        else:
            c = 1 / (1 + tf.exp(-self.k))
        return c


class SquaredKrausMap(KrausMap):
    def __init__(
        self,
        d=None,
        rank=None,
        spam=None,
        trainable=True,
        generate=True,
    ):

        KrausMap.__init__(d, rank, spam, trainable, generate=generate)

    def apply_channel(self, state):
        state = KrausMap.apply_channel(state)
        state = KrausMap.apply_channel(state)

        return state

    @property
    def choi(self):
        return channel_to_choi(self)


class TwoLocalKrausMap(KrausMap):
    def __init__(
        self,
        d=None,
        spam=None,
        trainable=True,
        generate=True,
    ):

        self.d = d
        self.n = int(np.log2(d))
        self.I = tf.eye(2, dtype=precision)
        self.kraus_list = [KrausMap(4, 16) for i in range(self.n - 1)]
        self.parameter_list = []
        for kraus in self.kraus_list:
            self.parameter_list.extend(kraus.parameter_list)

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        self.kraus = []
        for i, kraus in enumerate(self.kraus_list):
            kraus.generate_channel()
            I_start = i * [self.I]
            I_end = (self.n - i - 2) * [self.I]
            operators = I_start + [kraus.kraus] + I_end
            self.kraus.append(kron(*operators) / np.sqrt(self.n - 1))

        self.kraus = tf.concat(self.kraus, axis=1)
