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

    fidelity = tf.linalg.trace(sqrtm(C))
    return tf.abs(fidelity)**2


def channel_fidelity(map_A, map_B):
    choi_A = maps_to_choi([map_A])
    choi_B = maps_to_choi([map_B])
    d_squared = choi_A.shape[0]
    fidelity = state_fidelity(choi_A, choi_B)/d_squared

    return fidelity


def maps_to_choi(map_list):
    d = map_list[0].d
    choi = tf.zeros((d**2, d**2), dtype=precision)
    M = np.zeros((d**2, d, d))
    for i in range(d):
        for j in range(d):

            M[d*i + j, i, j] = 1

    M = tf.convert_to_tensor(M, dtype=precision)
    M_prime = tf.identity(M)
    for map in map_list:
        M_prime = map.apply_map(M_prime)
    for i in range(d**2):
        choi += tf.experimental.numpy.kron(M_prime[i], M[i])

    return choi


def expectation_value(probs, observable):
    ev = tf.abs(tf.reduce_sum(probs*observable, axis = 1))
    return ev


#@profile
def generate_ginibre(dim1, dim2, trainable = False):
    A = tf.cast(tf.random.normal((dim1, dim2), 0, 1), dtype = precision)
    B = tf.cast(tf.random.normal((dim1, dim2), 0, 1), dtype = precision)
    if trainable:
        A = tf.Variable(A, trainable = True)
        B = tf.Variable(B, trainable = True)

    X = A + 1j*B
    return X, A, B


def generate_state(d, rank):
    X, _, _ = generate_ginibre(d, rank)

    state = X@X.conj().T/np.trace(X@X.conj().T)
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


def apply_unitary(state, U):
    Ustate = tf.matmul(U, state)
    state = tf.matmul(Ustate, U, adjoint_b=True)
    return state


def variational_circuit(n, L):
    theta_list = [np.random.uniform(-np.pi, np.pi, 2*n) for i in range(L)]
    circuit = qk.QuantumCircuit(n)
    for theta in theta_list:
        for i, angle in enumerate(theta[:n]):
            circuit.ry(angle, i)

        for i, angle in enumerate(theta[n:]):
            circuit.rz(angle, i)

        for i in range(n-1):
            circuit.cnot(i, i+1)

    return circuit

