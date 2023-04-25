import numpy as np
import tensorflow as tf
from copy import copy, deepcopy
from scipy.linalg import expm
from utils import *


# def glue_operators(operator_list, q_list):
#    d_list = []
#    operator_full = tf.ones((1, 1))
#    for i, (operator, q) in enumerate(zip(operator_list, q_list)):
#        d_full = operator_full.shape[0]
#        d = 2**q / d_full
#        if d < 1:
#            raise ValueError("qubit numbers are not valid")
#        operator_full = kron(operator_full, tf.eye(d))
#        operator_full = kron(operator_full, operator)


def swap_unitary(n, q1, q2):
    zero_zero = tf.convert_to_tensor([[1, 0], [0, 0]], dtype=tf.complex128)
    zero_one = tf.convert_to_tensor([[0, 1], [0, 0]], dtype=tf.complex128)
    one_zero = tf.convert_to_tensor([[0, 0], [1, 0]], dtype=tf.complex128)
    one_one = tf.convert_to_tensor([[0, 0], [0, 1]], dtype=tf.complex128)

    d1 = 2**q1
    d2 = 2 ** (q2 - q1 - 1)
    d3 = 2 ** (n - q2 - 1)

    I_d1 = tf.eye(d1, dtype=tf.complex128)
    I_d2 = tf.eye(d2, dtype=tf.complex128)
    I_d3 = tf.eye(d3, dtype=tf.complex128)

    zero_zero_full = kron(I_d1, zero_zero, I_d2, zero_zero, I_d3)
    zero_one_full = kron(I_d1, zero_one, I_d2, one_zero, I_d3)
    one_zero_full = kron(I_d1, one_zero, I_d2, zero_one, I_d3)
    one_one_full = kron(I_d1, one_one, I_d2, one_one, I_d3)

    U = zero_zero_full + zero_one_full + one_zero_full + one_one_full

    return U


def pad_unitary(U, n, target_qubits):
    if not isinstance(target_qubits, list):
        target_qubits = [target_qubits]
    d = U.shape[0]
    num_qubits = int(np.log2(d))

    if num_qubits != len(target_qubits):
        raise ValueError("Gate and target qubits are incompatible")

    q1 = target_qubits[0]
    d1 = 2**q1
    d2 = 2 ** (n - q1 - len(target_qubits))

    U = kron(tf.eye(d1), U)
    U = kron(U, tf.eye(d2))

    if num_qubits > 1:
        q2 = target_qubits[1]
        if q2 != q1 + 1:
            U_swap = swap_unitary(n, q1 + 1, q2)
            U = U_swap @ U @ U_swap

    return U


class Gate:
    def get_unitary(self, n):
        raise NotImplementedError

    def __call__(self, state):
        U = self.get_unitary()
        return state @ U


class CNOT(Gate):
    def __init__(self, n, target_qubits):
        self.n = n
        self.target_qubits = target_qubits

    def get_unitary(self):
        U = tf.convert_to_tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=tf.complex128,
        )
        U = pad_unitary(U, self.n, self.target_qubits)
        return U

    def __call__(self, state):
        U = tf.convert_to_tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=tf.complex128,
        )

        state = tf.tensordot(U, state, axes=[[2, 3], self.target_qubits])
        state = tf.moveaxis(state, [0, 1], self.target_qubits)
        return state @ U


class Ry(Gate):
    def __init__(self, theta, n, target_qubits):
        self.theta = theta
        self.n = n
        self.target_qubits = target_qubits

        if not isinstance(self.target_qubits, list):
            self.target_qubits = [self.target_qubits]

        self.I = tf.cast([[[1.0, 0.0], [0.0, 1.0]]], dtype=tf.complex128)
        self.Y = tf.cast([[[0.0, -1j], [1j, 0.0]]], dtype=tf.complex128)

    def get_unitary(self):
        U = tf.cos(self.theta / 2) * self.I + 1j * tf.sin(self.theta / 2) * self.Y
        U = pad_unitary(U, self.n, self.target_qubits)
        return U


class Rx(Gate):
    def __init__(self, theta, n, target_qubits):
        self.theta = theta
        self.n = n
        if not isinstance(target_qubits, list):
            target_qubits = [target_qubits]
        self.target_qubits = target_qubits

        self.I = tf.cast([[[1, 0], [0, 1]]], dtype=tf.complex128)
        self.X = tf.cast([[[0, 1], [1, 0]]], dtype=tf.complex128)

    def get_unitary(self):
        U = tf.cos(self.theta / 2) * self.I + 1j * tf.sin(self.theta / 2) * self.X
        U = pad_unitary(U, self.n, self.target_qubits)
        return U

    def __call__(self, state):
        U = self.get_unitary()
        return U @ state


class LadderCNOT(Gate):
    def __init__(self, n):
        self.n = n

    def get_unitary(self):
        U = np.eye(2**self.n).reshape(1, -1)
        U = tf.convert_to_tensor(U, dtype=tf.complex128)
        for i in range(self.n - 1):
            cnot = CNOT(self.n, [i, i + 1])
            U = U @ cnot.get_unitary()
        return U


class LadderRy(Gate):
    def __init__(self, theta_list, n):
        self.theta_list = theta_list
        self.n = n

    def get_unitary(self):
        U = np.eye(2**self.n).reshape(1, -1)
        U = tf.convert_to_tensor(U, dtype=tf.complex128)
        for i in range(self.n):
            theta = tf.cast(self.theta_list[i], tf.complex128)
            ry = Ry(theta, self.n, i)
            U = U @ ry.get_unitary()
        return U


class LadderRx(Gate):
    def __init__(self, theta_list, n):
        self.theta_list = theta_list
        self.n = n

    def get_unitary(self):
        U = np.eye(2**self.n).reshape(1, -1)
        U = tf.convert_to_tensor(U, dtype=tf.complex128)
        for i in range(self.n):
            theta = tf.cast(self.theta_list[i], tf.complex128)
            rx = Rx(theta, self.n, i)
            U = U @ rx.get_unitary()
        return U


class QNN:
    def __init__(self, n, num_layers):
        self.n = n
        self.num_layers = num_layers

        self.d = 2**n

        self.params = []
        for i in range(num_layers):
            theta_list = tf.Variable(tf.random.normal([n], dtype=tf.float64))
            self.params.append(theta_list)

    def __call__(self, inputs):
        batch_dim = inputs.shape[0]
        state = np.zeros((1, self.d))
        state[0, 0] = 1
        state = tf.convert_to_tensor(state, dtype=tf.complex128)
        state = tf.repeat(state, batch_dim, axis=0)

        cnotLadder = LadderCNOT(self.n)
        for i in range(self.num_layers):
            ry = LadderRy(inputs, self.n)
            rx = LadderRx(self.params[i], self.n)
            state = ry(state)
            state = rx(state)
            if i != self.num_layers - 1:
                state = cnotLadder(state)

        return tf.abs(state) ** 2
