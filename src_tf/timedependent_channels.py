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


class Liouvillian():
    def __init__(self,)
        pass

    def __call__(self, t):
        pass

class SpinSpin(Liouvillian):
    def __init__(self, u):
        self.d = 4
        self.rank = 1
        self.u = tf.Variable(u, trainable=True)
        #self.x = tf.Variable(0, trainable=True)
        #self.y = tf.Variable(0, trainable=True)
        #self.z = tf.Variable(0, trainable=True)

        self.parameter_list = [self.u]

    def __call__(self, t):
        X = tf.Variable([[0, 1], [1, 0]], dtype=precision)
        Y = tf.Variable([[0, -1j], [1j, 0]], dtype=precision)
        Z = tf.Variable([[1, 0], [0, -1]], dtype=precision)

        H = kron(X,X) + kron(Y,Y) + kron(Z,Z)
        H = tf.cast(self.u, precision)*H

        I = tf.eye(self.d, dtype=precision)
        HH = (kron(H, I) - kron(I, tf.transpose(H)))
        HH = tf.repeat(HH[None, :, :], len(t), axis=0)
        L = -1j*HH + 0.5*()
        return L


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
        self.grid_size = grid_size

        self.d = liouvillian.d
        self.rank = liouvillian.rank
        self.parameter_list = liouvillian.parameter_list

        if spam is None:
            spam = IdealSPAM(self.d)
        self.spam = spam

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        pass

    def apply_channel(self, state):

        state, t = state
        T = self.super_operator(t)
        
        state = tf.reshape(state, (self.d**2, 1))
        state = tf.linalg.matmul(T, state)
        state = tf.reshape(state, (self.d, self.d))
        return state

    def super_operator(self, t):
        t_list = tf.linspace(0., t, self.grid_size)
        dt = t/self.grid_size

        L = self.liouvillian(t_list)
        eL = tf.linalg.expm(dt*L)
        T = tf.eye(self.d**2, dtype=precision)
        
        for i in range(self.grid_size):
            T = tf.linalg.matmul(T, eL[i])

        return T

    def choi(self, t):
        return reshuffle(self.super_operator(t))


