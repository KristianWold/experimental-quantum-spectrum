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


def partial_trace(X, discard_first = True):
    d = int(np.sqrt(X.shape[0]))
    X = tf.reshape(X, (d,d,d,d))
    if discard_first:
        Y = tf.einsum("ijik->jk", X)
    else:
        Y = tf.einsum("jiki->jk", X)
    return Y


def state_fidelity(A, B):

    sqrtB = tf.linalg.sqrtm(B)
    C = sqrtB@A@sqrtB

    sqrtC = tf.linalg.sqrtm(C)
    fidelity = tf.linalg.trace(sqrtC)
    return tf.abs(fidelity)**2


def expectation_value(probs, observable):
    ev = tf.abs(tf.reduce_sum(probs*observable, axis = 1))
    return ev


#@profile
def generate_ginibre(dim1, dim2, trainable = False, complex = True):
    A = tf.random.normal((dim1, dim2), 0, 1)
    if complex:
        B = tf.random.normal((dim1, dim2), 0, 1)
    else:
        B = None
    if trainable:
        A = tf.Variable(A, trainable = True)
        if B is not None:
            B = tf.Variable(B, trainable = True)

    X = A
    if complex:
        X = tf.cast(X, dtype=precision) + 1j*tf.cast(B, dtype=precision)
    return X, A, B


def generate_state(d, rank):
    X, _, _ = generate_ginibre(d, rank)

    XX = tf.linalg.matmul(X,X, adjoint_b = True)
    state = XX/tf.linalg.trace(XX)
    return state


def generate_unitary(d=None, G=None):
    if G is None:
        G, _, _ = generate_ginibre(d, d)
    Q, R = tf.linalg.qr(G, full_matrices = False)
    D = tf.linalg.tensor_diag_part(R)
    D = tf.math.sign(D)
    D = tf.linalg.diag(D)
    U = Q@D

    return U


def circuit_to_matrix(circuit):
    U = Operator(circuit.reverse_bits()).data
    U = tf.convert_to_tensor(U, dtype = precision)

    return U


def channel_to_choi(channel_list):
    d = channel_list[0].d
    choi = tf.zeros((d**2, d**2), dtype=precision)
    M = np.zeros((d**2, d, d))
    for i in range(d):
        for j in range(d):
            M[d*i + j, i, j] = 1

    M = tf.convert_to_tensor(M, dtype=precision)
    M_prime = tf.identity(M)
    for channel in channel_list:
        M_prime = channel.apply_channel(M_prime)

    for i in range(d**2):
        choi += tf.experimental.numpy.kron(M_prime[i], M[i])

    return choi


def channel_fidelity(map_A, map_B):
    choi_A = maps_to_choi([map_A])
    choi_B = maps_to_choi([map_B])
    d_squared = choi_A.shape[0]
    fidelity = state_fidelity(choi_A, choi_B)/d_squared

    return fidelity


def apply_unitary(state, U):
    Ustate = tf.matmul(U, state)
    state = tf.matmul(Ustate, U, adjoint_b=True)
    return state


def attraction(channel, N=1000):
    d = channel.d
    I = tf.cast(tf.eye(d, batch_shape=(N,)), dtype = precision)/d

    state_list = []
    state = np.zeros((d,d))
    state[0,0] = 1
    for i in range(N):
        U = random_unitary(d).data
        state_haar = DensityMatrix(U@state@U.T.conj()).data
        state_list.append(state_haar)
    
    state = tf.convert_to_tensor(state_list)

    state = channel.apply_channel(state)
    att = tf.math.reduce_mean(state_fidelity(state, I))

    return att


def measurement(state, U_basis=None, povm=None):
    d = state.shape[1]
    if U_basis is None:
        U_basis = tf.eye(d, dtype=precision)
        U_basis = tf.expand_dims(U_basis, axis = 0)

    if povm is None:
        povm = corr_mat_to_povm(np.eye(d))

    Ustate = tf.matmul(U_basis, state)
    UstateU = tf.matmul(Ustate, U_basis, adjoint_b=True)

    state = tf.expand_dims(UstateU, axis=1)
    povm = tf.expand_dims(povm, axis=0)

    probs = tf.linalg.trace(state@povm)

    return probs


def add_noise_to_probs(tensor, sigma = 0.01):
    tensor = tensor + tf.math.sqrt(tensor*(1-tensor))*tf.cast(tf.random.normal(tensor.shape, 0, sigma), dtype = precision)
    tensor = tensor/tf.math.reduce_sum(tensor, axis=1, keepdims=True)

    return tensor

        
