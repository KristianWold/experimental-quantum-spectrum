import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import tensorflow as tf

from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm


def numberToBase(n, b, num_digits):
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b

    while len(digits)<num_digits:
        digits.append(0)
    return digits[::-1]


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
    fidelity = tf.linalg.trace(tf.linalg.sqrtm(sqrtB@A@sqrtB))
    return fidelity


def apply_map(state, choi):
    d = state.shape[0]

    #reshuffle
    choi = tf.reshape(choi, (d,d,d,d))
    choi = tf.transpose(choi, perm = [0,2,1,3])
    choi = tf.reshape(choi, (d**2,d**2))

    #flatten
    state = tf.reshape(state, (d**2, 1))

    state = tf.reshape(choi@state, (d, d))
    return state


def prepare_input(config, return_unitary = False):
    """1 = |0>, 2 = |1>, 3 = |+>, 4 = |->, 5 = |+i>, 6 = |-i>"""
    n = len(config)
    circuit = qk.QuantumCircuit(n)
    for i, gate in enumerate(config):
        if gate == 0:
            circuit.i(i)
        if gate == 1:
            circuit.x(i)
        if gate == 2:
            circuit.h(i)
        if gate == 3:
            circuit.x(i)
            circuit.h(i)
        if gate == 4:
            circuit.h(i)
            circuit.s(i)
        if gate == 5:
            circuit.x(i)
            circuit.h(i)
            circuit.s(i)

    if not return_unitary:
        state = tf.convert_to_tensor(DensityMatrix(circuit).data, dtype=tf.complex128)
    else:
        state = Operator(circuit).data

    return state


def bitstring_density(state, basis):

    return np.abs(np.diag(basis.conj().T@state@basis))


def generate_ginibre(dim1, dim2, real = False):
    ginibre = np.random.normal(0, 1, (dim1, dim2))
    if not real:
         ginibre = ginibre + 1j*np.random.normal(0, 1, (dim1, dim2))
    return tf.convert_to_tensor(ginibre, dtype=tf.complex128)


def generate_state(dim1, dim2):
    X = generate_ginibre(dim1, dim2)

    state = X@X.conj().T/torch.trace(X@X.conj().T)
    return state


def generate_choi(X):
    d = int(np.sqrt(X.shape[0]))  # dim of Hilbert space
    I = tf.eye(d, dtype = tf.complex128)
    XX = X@tf.linalg.adjoint(X)

    #partial trace
    Y = tf.linalg.sqrtm(tf.linalg.inv(partial_trace(XX)))
    Ykron = tf.experimental.numpy.kron(I, Y)

    #choi
    choi = Ykron@XX@Ykron

    return choi
