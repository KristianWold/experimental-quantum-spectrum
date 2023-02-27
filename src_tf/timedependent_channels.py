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


class Hamiltonian():
    def __init__(self, d, rank, trainable=True):
        self.d = d

    def __call__(self, t):
        


class MagnusPropagator(Channel):
    def __init__(
        self, 
        liouvillian = None,
        spam=None,
        trainable=True,
        generate=True,
        grid_size=100,
    ):
        self.liouvillian = liouvillian
        self.d = liouvillian.d
        self.rank = liouvillian.rank
        self.grid_size = grid_size

        if spam is None:
            spam = IdealSPAM(d)
        self.spam = spam

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        pass

    def apply_channel(self, state):

        state, t = state
        t_list = tf.linspace(0, t, self.grid_size)
        L = self.liouvillian(t_list)
        eL = tf.linalg.expm(L)
        T = tf.eye(self.d**2, dtype=precision)
        for i in range(self.grid_size):
            T = tf.linalg.matmul(T, eL[i])

        return state

    @property
    def choi(self):
        return reshuffle(self.super_operator)
