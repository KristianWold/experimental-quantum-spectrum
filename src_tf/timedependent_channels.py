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
from optimization import *


class Liouvillian:
    def __init__(self):
        pass

    def __call__(self, t):
        pass


class Hamiltonian(Liouvillian):
    def __init__(
        self,
    ):
        self.d = 4
        self.rank = 1
        self.u1 = tf.Variable(tf.random.normal(0, 1), trainable=True)
        self.u2 = tf.Variable(tf.random.normal(0, 1), trainable=True)
        self.u3 = tf.Variable(tf.random.normal(0, 1), trainable=True)

        self.parameter_list = [self.u1, self.u2, self.u3]

    def __call__(self, t):
        X = tf.Variable([[0, 1], [1, 0]], dtype=precision)

        Y = tf.Variable([[0, -1j], [1j, 0]], dtype=precision)
        Z = tf.Variable([[1, 0], [0, -1]], dtype=precision)

        H = self.u1 * kron(X, X) + self.u2 * kron(Y, Y) + self.u3 * kron(Z, Z)
        H = tf.cast(H, precision)

        I = tf.eye(self.d, dtype=precision)
        HH = kron(H, I) - kron(I, tf.transpose(H))
        HH = tf.repeat(HH[None, :, :], t.shape, axis=0)
        L = -1j * HH + 0.5 * ()
        return L


class SpinSpin(Liouvillian):
    def __init__(self, degree=3):
        self.degree = degree
        self.d = 4
        self.rank = 1
        self.I = tf.eye(2, dtype=precision)

        self.u = tf.Variable(tf.random.normal([3], 0, 1), trainable=True)
        self.theta_sin = None
        if degree > 0:
            self.theta_sin = tf.Variable(
                tf.random.normal([6 * self.degree], 0, 1), trainable=True
            )
        self.theta_cos = tf.Variable(
            tf.random.normal([6 * (self.degree + 1)], 0, 1), trainable=True
        )

        self.parameter_list = [
            self.u,
            self.theta_cos,
        ]
        if self.theta_sin is not None:
            self.parameter_list.append(self.theta_sin)

    def __call__(self, t):
        t = tf.cast(t, precision)[:, tf.newaxis, tf.newaxis]
        u = tf.cast(self.u, precision)
        if self.theta_sin is not None:
            theta_sin = tf.cast(self.theta_sin, precision)
        theta_cos = tf.cast(self.theta_cos, precision)

        X = tf.convert_to_tensor([[0, 1], [1, 0]], dtype=precision)
        Y = tf.convert_to_tensor([[0, -1j], [1j, 0]], dtype=precision)
        Z = tf.convert_to_tensor([[1, 0], [0, -1]], dtype=precision)

        XX = kron(X, X)
        YY = kron(Y, Y)
        ZZ = kron(Z, Z)
        XI = kron(X, self.I)
        YI = kron(Y, self.I)
        ZI = kron(Z, self.I)
        IX = kron(self.I, X)
        IY = kron(self.I, Y)
        IZ = kron(self.I, Z)

        XX = tf.repeat(XX[None, :, :], t.shape, axis=0)
        YY = tf.repeat(YY[None, :, :], t.shape, axis=0)
        ZZ = tf.repeat(ZZ[None, :, :], t.shape, axis=0)
        XI = tf.repeat(XI[None, :, :], t.shape, axis=0)
        YI = tf.repeat(YI[None, :, :], t.shape, axis=0)
        ZI = tf.repeat(ZI[None, :, :], t.shape, axis=0)
        IX = tf.repeat(IX[None, :, :], t.shape, axis=0)
        IY = tf.repeat(IY[None, :, :], t.shape, axis=0)
        IZ = tf.repeat(IZ[None, :, :], t.shape, axis=0)

        H = 0.0
        for j in range(self.degree + 1):
            H += (
                theta_cos[j] * tf.math.cos(2 * np.pi * j * t) * XI
                + theta_cos[self.degree + j] * tf.math.cos(2 * np.pi * j * t) * YI
                + theta_cos[2 * self.degree + j] * tf.math.cos(2 * np.pi * j * t) * ZI
                + theta_cos[3 * self.degree + j] * tf.math.cos(2 * np.pi * j * t) * IX
                + theta_cos[4 * self.degree + j] * tf.math.cos(2 * np.pi * j * t) * IY
                + theta_cos[5 * self.degree + j] * tf.math.cos(2 * np.pi * j * t) * IZ
            )
        for j in range(self.degree):
            H += (
                theta_sin[j] * tf.math.sin(2 * np.pi * (j + 1) * t) * XI
                + theta_sin[(self.degree - 1) + j]
                * tf.math.sin(2 * np.pi * (j + 1) * t)
                * YI
                + theta_sin[2 * (self.degree - 1) + j]
                * tf.math.sin(2 * np.pi * (j + 1) * t)
                * ZI
                + theta_sin[3 * (self.degree - 1) + j]
                * tf.math.sin(2 * np.pi * (j + 1) * t)
                * IX
                + theta_sin[4 * (self.degree - 1) + j]
                * tf.math.sin(2 * np.pi * (j + 1) * t)
                * IY
                + theta_sin[5 * (self.degree - 1) + j]
                * tf.math.sin(2 * np.pi * (j + 1) * t)
                * IZ
            )
        H += u[0] * XX + u[1] * YY + u[2] * ZZ

        return H

    def generate_Fourier_series(self):
        if self.theta_sin is not None:
            theta_sin = self.theta_sin.numpy().reshape(6, -1)
        else:
            theta_sin = np.zeros((6, self.degree))

        theta_cos = self.theta_cos.numpy().reshape(6, -1)

        signal_list = [0, 0, 0, 0, 0, 0]
        t = np.linspace(0, 1, 1000)
        for idx in range(6):
            for i, theta in enumerate(zip(theta_sin[idx])):
                signal_list[idx] += theta * np.sin(2 * np.pi * (i + 1) * t)

            for i, theta in enumerate(zip(theta_cos[idx])):
                signal_list[idx] += theta * np.cos(2 * np.pi * i * t)

        return t, signal_list


class PolynomialHamiltonian:
    def __init__(self, d=None, degree=None):
        self.d = d
        self.degree = degree

        self.A = tf.Variable(
            tf.random.normal([degree + 1, d, d], 0, 1, dtype=tf.float64), trainable=True
        )
        self.B = tf.Variable(
            tf.random.normal([degree + 1, d, d], 0, 1, dtype=tf.float64), trainable=True
        )

        self.parameter_list = [self.A, self.B]

    def __call__(self, t):
        t = tf.cast(t, precision)[:, tf.newaxis, tf.newaxis]
        G = tf.complex(self.A, self.B)
        H = G + tf.linalg.adjoint(G)
        H_ = 0
        for i in range(self.degree + 1):
            H_ += H[tf.newaxis, i, :, :] * t**i

        return H_


class JumpOperator:
    def __init__(self, d, trainable=False, trace_less=True):
        self.d = d
        self.trace_less = trace_less
        self.A = tf.random.normal([d, d], 0, 1, dtype=tf.float64)
        self.B = tf.random.normal([d, d], 0, 1, dtype=tf.float64)
        self.parameter_list = []

        if trainable:
            self.A = tf.Variable(self.A, trainable=True)
            self.B = tf.Variable(self.B, trainable=True)
            self.parameter_list.extend([self.A, self.B])

    def __call__(self, t):
        J = tf.complex(self.A, self.B)
        if self.trace_less:
            J = J - tf.linalg.trace(J) * tf.eye(self.d, dtype=precision) / self.d
        norm = tf.linalg.trace(tf.matmul(J, J, adjoint_a=True))
        J = J / tf.math.sqrt(norm)
        J = tf.repeat(J[None, :, :], t.shape, axis=0)
        return J

    def set_trainable(self, trainable):
        self.A = tf.Variable(self.A, trainable=trainable)
        self.B = tf.Variable(self.B, trainable=trainable)
        self.parameter_list = []
        if trainable:
            self.parameter_list = [self.A, self.B]


class LindbladGenerator:
    def __init__(self, hamiltonian, jump_operator, gamma=0):
        self.Hamiltonian = hamiltonian
        self.JumpOperator = jump_operator
        self.gamma = gamma

        self.d = hamiltonian.d
        self.parameter_list = hamiltonian.parameter_list
        if jump_operator is not None:
            self.parameter_list.extend(jump_operator.parameter_list)

    def __call__(self, t):
        H = self.Hamiltonian(t)
        I = tf.repeat(tf.eye(self.d, dtype=precision)[None, :, :], t.shape, axis=0)

        L = -1j * (tf_kron(H, I) - tf_kron(I, tf.transpose(H, [0, 2, 1])))

        if self.gamma != 0:
            J = self.JumpOperator(t)
            JJ = tf.matmul(J, J, adjoint_a=True)
            L += self.gamma * (
                tf_kron(J, tf.math.conj(J))
                - 0.5 * tf_kron(JJ, I)
                - 0.5 * tf_kron(I, tf.math.conj(JJ))
            )

        return L


class MagnusPropagator(Channel):
    def __init__(
        self,
        liouvillian=None,
        spam=None,
        trainable=True,
        generate=True,
        grid_size=100,
        T=1,
    ):
        self.liouvillian = liouvillian
        self.grid_size = grid_size
        self.T = T

        self.d = liouvillian.d
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
        dt = t / (self.grid_size - 1)
        t_list = tf.linspace(dt / 2, t - dt / 2, self.grid_size - 1)

        L = self.liouvillian(t_list)
        eL = tf.linalg.expm(dt * L)
        T = tf.eye(self.d**2, dtype=precision)

        for i in range(self.grid_size - 1):
            T = tf.linalg.matmul(T, eL[i])

        return T

    @property
    def choi(self):
        return reshuffle(self.super_operator(self.T))

    def get_choi(self, T):
        return reshuffle(self.super_operator(T))
    


def fit_spinspin(
    channel_target=None, degree=0, gamma=0, filename=None, grid_size=200, num_iter=500
):
    jump_operator = JumpOperator(4, trainable=False)
    H_model = SpinSpin(degree=degree)

    lindblad_model = LindbladGenerator(
        hamiltonian=H_model,
        jump_operator=jump_operator,
        gamma=gamma,
    )

    channel_model = MagnusPropagator(
        liouvillian=lindblad_model, grid_size=grid_size, T=1
    )

    model = ModelQuantumMap(
        channel=channel_model,
        loss_function=channel_mse_loss,
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        logger=Logger(loss_function=channel_fidelity_loss, sample_freq=1, N=0),
    )

    model.train(
        inputs=None,
        targets=[channel_target],
        num_iter=num_iter,
        N=0,
        verbose=False,
    )
    model.optimizer = None

    saver(model, filename + ".model")
