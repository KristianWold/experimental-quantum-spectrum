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


class StinespringHamiltonianMap(Channel):
    def __init__(
        self,
        d=None,
        logrank=None,
        spam=None,
        trainable=True,
        generate=True,
    ):

        self.d = d
        self.rank = 2**logrank

        if spam is None:
            spam = IdealSPAM(d=d)
        self.spam = spam

        _, self.A, self.B = generate_ginibre(
            self.rank * d, self.rank * d, trainable=trainable
        )
        self.parameter_list = [self.A, self.B]

        self.stinespring = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.cast(self.A, dtype=precision) + 1j * tf.cast(self.B, dtype=precision)
        self.H = tf.expand_dims(tf.matmul(G, G, adjoint_b=True), axis=0)

    def apply_channel(self, state):
        state, t = state
        t = t[:, tf.newaxis, tf.newaxis]
        U = tf.linalg.expm(-1j * t * self.H)[:, :, : self.d]
        Ustate = tf.matmul(U, state)
        UstateU = tf.matmul(Ustate, U, adjoint_b=True)
        state = partial_trace(UstateU)

        return state

    @property
    def choi(self):
        choi = tf.experimental.numpy.kron(self.U, tf.math.conj(self.U))
        return choi
