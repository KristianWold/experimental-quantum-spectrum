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

