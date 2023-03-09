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
    def __init__(self):
        pass

    def __call__(self, t):
        pass


class Hamiltonian(Liouvillian):
    def __init__(self,):
        self.d = 4
        self.rank = 1
        self.u1 = tf.Variable(tf.random.normal(0,1), trainable=True)
        self.u2 = tf.Variable(tf.random.normal(0,1), trainable=True)
        self.u3 = tf.Variable(tf.random.normal(0,1), trainable=True)

        self.parameter_list = [self.u1, self.u2, self.u3]

    def __call__(self, t):
        X = tf.Variable([[0, 1], [1, 0]], dtype=precision)
        
        Y = tf.Variable([[0, -1j], [1j, 0]], dtype=precision)
        Z = tf.Variable([[1, 0], [0, -1]], dtype=precision)

        H = self.u1*kron(X,X) + self.u2*kron(Y,Y) + self.u3*kron(Z,Z)
        H = tf.cast(H, precision)

        I = tf.eye(self.d, dtype=precision)
        HH = (kron(H, I) - kron(I, tf.transpose(H)))
        HH = tf.repeat(HH[None, :, :], len(t), axis=0)
        L = -1j*HH + 0.5*()
        return L


class SpinSpin(Liouvillian):
    def __init__(self):
        self.d = 4
        self.rank = 1
        self.I = tf.eye(2, dtype=precision)
        self.u1 = tf.Variable(tf.random.normal([], 0, 1), trainable=True)
        self.u2 = tf.Variable(tf.random.normal([], 0, 1), trainable=True)
        self.u3 = tf.Variable(tf.random.normal([], 0, 1), trainable=True)
        

        self.degree = 3
        self.theta1_x = tf.Variable(tf.random.normal([self.degree], 0, 1), trainable=True)
        self.theta1_y = tf.Variable(tf.random.normal([self.degree], 0, 1), trainable=True)
        self.theta2_x = tf.Variable(tf.random.normal([self.degree], 0, 1), trainable=True)
        self.theta2_y = tf.Variable(tf.random.normal([self.degree], 0, 1), trainable=True)

        self.parameter_list = [self.u1, self.u2, self.u3, self.theta1_x, self.theta1_y, self.theta2_x, self.theta2_y]

    def __call__(self, t):
        u1 = tf.cast(self.u1, precision)
        u2 = tf.cast(self.u2, precision)
        u3 = tf.cast(self.u3, precision)
        theta1_x = tf.cast(self.theta1_x, precision)
        theta1_y = tf.cast(self.theta1_y, precision)
        theta2_x = tf.cast(self.theta2_x, precision)
        theta2_y = tf.cast(self.theta2_y, precision)

        X = tf.Variable([[0, 1], [1, 0]], dtype=precision)
        Y = tf.Variable([[0, -1j], [1j, 0]], dtype=precision)
        Z = tf.Variable([[1, 0], [0, -1]], dtype=precision)
        
        XX = kron(X, X)
        YY = kron(Y, Y)
        ZZ = kron(Z, Z)

        XI = kron(X, self.I)
        YI = kron(Y, self.I)

        IX = kron(self.I, X)
        IY = kron(self.I, Y)

        L_list = []
        for i, t_ in enumerate(t):
            H = 0.
            t_ = tf.cast(t_, precision)
            for j in range(self.degree):
                H += theta1_x[j]*tf.math.cos((j+1)*t_)*XI + theta1_y[j]*tf.math.cos((j+1)*t_)*YI
                H += theta2_x[j]*tf.math.cos((j+1)*t_)*IX + theta2_y[j]*tf.math.cos((j+1)*t_)*IY
            H += u1*kron(X,X) + u2*kron(Y,Y) + u3*kron(Z,Z)


            I = tf.eye(self.d, dtype=precision)
            HH = (kron(H, I) - kron(I, tf.transpose(H)))
            L = -1j*HH
            L_list.append(L)
            
        L = tf.stack(L_list, axis=0)
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
        dt = t/(self.grid_size-1)
        t_list = tf.linspace(dt/2, t - dt/2, self.grid_size-1)

        L = self.liouvillian(t_list)
        eL = tf.linalg.expm(dt*L)
        T = tf.eye(self.d**2, dtype=precision)
        
        for i in range(self.grid_size-1):
            T = tf.linalg.matmul(T, eL[i])

        return T

    def choi(self, t):
        return reshuffle(self.super_operator(t))


