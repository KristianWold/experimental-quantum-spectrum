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

from utils import *
from set_precision import *


def partial_trace(state, discard_first=True):
    d = int(np.sqrt(state.shape[1]))
    state = tf.reshape(state, (-1, d, d, d, d))
    if discard_first:
        state = tf.einsum("bijik->bjk", state)
    else:
        state = tf.einsum("bjiki->bjk", state)
    return state


def partial_transpose(state, qubit):
    d = state.shape[1]
    n = int(np.log2(d))
    shape = 2 * n * [2]
    state = tf.reshape(state, shape)
    new_shape = list(range(2 * n))
    new_shape[qubit] = qubit + n
    new_shape[qubit + n] = qubit
    state = tf.transpose(state, perm=new_shape)
    state = tf.reshape(state, (1, d, d))

    return state


def state_fidelity(A, B):

    sqrtB = tf.linalg.sqrtm(B)
    C = sqrtB @ A @ sqrtB

    sqrtC = tf.linalg.sqrtm(C)
    fidelity = tf.linalg.trace(sqrtC)
    return tf.abs(fidelity) ** 2


def expectation_value(probs, observable):
    ev = tf.abs(tf.reduce_sum(probs * observable, axis=1))
    return ev


# @profile
def generate_ginibre(dim1, dim2, trainable=False, complex=True):
    A = tf.random.normal((dim1, dim2), 0, 1, dtype=tf.float64)

    if complex:
        B = tf.random.normal((dim1, dim2), 0, 1, dtype=tf.float64)
    else:
        B = None
    if trainable:
        A = tf.Variable(A, trainable=True)
        if B is not None:
            B = tf.Variable(B, trainable=True)

    X = A
    if complex:
        X = tf.cast(X, dtype=precision) + 1j * tf.cast(B, dtype=precision)
    return X, A, B


def generate_state(d, rank):
    X, _, _ = generate_ginibre(d, rank)

    XX = tf.linalg.matmul(X, X, adjoint_b=True)
    state = XX / tf.linalg.trace(XX)
    return state


def generate_unitary(d=None, G=None):
    if G is None:
        G, _, _ = generate_ginibre(d, d)
    Q, R = tf.linalg.qr(G, full_matrices=False)
    D = tf.linalg.tensor_diag_part(R)
    D = tf.math.sign(D)
    D = tf.linalg.diag(D)
    U = Q @ D

    return U


def circuit_to_matrix(circuit):
    U = Operator(circuit.reverse_bits()).data
    U = tf.convert_to_tensor(U, dtype=precision)

    return U


def channel_to_choi(channel_list):
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


def apply_unitary(state, U):
    Ustate = tf.matmul(U, state)
    UstateU = tf.matmul(Ustate, U, adjoint_b=True)
    return UstateU


def attraction(channel, N=1000):
    d = channel.d
    I = tf.cast(tf.eye(d, batch_shape=(N,)), dtype=precision) / d

    state_list = []
    state = np.zeros((d, d))
    state[0, 0] = 1
    for i in range(N):
        U = random_unitary(d).data
        state_haar = DensityMatrix(U @ state @ U.T.conj()).data
        state_list.append(state_haar)

    state = tf.convert_to_tensor(state_list)

    state = channel.apply_channel(state)
    att = tf.math.reduce_mean(state_fidelity(state, I))

    return att


def corr_mat_to_povm(corr_mat):
    d = corr_mat.shape[0]
    povm = []
    for i in range(d):
        M = tf.linalg.diag(corr_mat[i, :])
        povm.append(M)

    povm = tf.convert_to_tensor(povm, dtype=precision)

    return povm


def init_ideal(d):
    init = np.zeros((d, d))
    init[0, 0] = 1
    init = tf.convert_to_tensor(init, dtype=precision)
    return init


def povm_ideal(d):
    povm = tf.cast(corr_mat_to_povm(np.eye(d)), dtype=precision)
    return povm


def measurement(state, U_basis=None, povm=None):
    d = state.shape[1]

    if povm is None:
        povm = corr_mat_to_povm(np.eye(d))

    if U_basis is not None:
        state = apply_unitary(state, U_basis)

    state = tf.expand_dims(state, axis=1)
    povm = tf.expand_dims(povm, axis=0)

    probs = tf.linalg.trace(state @ povm)

    return probs


def add_noise_to_probs(tensor, noise=0.01):
    tensor = tensor + tf.math.sqrt(tensor * (1 - tensor)) * tf.cast(
        tf.random.normal(tensor.shape, 0, noise), dtype=precision
    )
    tensor = tensor / tf.math.reduce_sum(tensor, axis=1, keepdims=True)

    return tensor


def spectrum_to_radial(spectrum):
    radial = tf.norm(spectrum, axis=1)
    return radial


def spectrum_to_angular(spectrum):
    angular = tf.math.angle(spectrum[:-1, 0] + 1j * spectrum[:-1, 1])
    return angular
