import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm
from utils import *


def partial_trace(X, discard_first = True):
    d = int(np.sqrt(X.shape[0]))
    X = np.reshape(X, (d,d,d,d))
    if discard_first:
        Y = np.einsum("ijik->jk", X)
    else:
        Y = np.einsum("jiki->jk", X)
    return Y


def state_fidelity(A, B):

    sqrtB = sqrtm(B)
    C = sqrtB@A@sqrtB

    fidelity = np.trace(sqrtm(C))
    return np.abs(fidelity)**2


def state_norm(A, B):

    norm = 1 - np.linalg.norm(A - B)
    return np.abs(norm)


#@profile
def partial_trace(X, discard_first = True):
    d = int(np.sqrt(X.shape[0]))
    X = X.reshape(d,d,d,d)
    if discard_first:
        Y = np.einsum("ijik->jk", X)
    else:
        Y = np.einsum("jiki->jk", X)
    return Y


def expectation_value(probs, observable):
    ev = np.sum(probs*observable)
    return ev


def measurement(state, U_basis, povm):
    state = U_basis@state@U_basis.T.conj()
    state = sum([M@state@M.T.conj() for M in povm])
    probs = np.diag(state)
    return probs


def parity_observable(n, trace_index_list=[]):
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    observable = n*[Z]
    for index in trace_index_list:
        observable[index] = I

    observable = np.diag(kron(*observable))
    return observable

#@profile
def generate_ginibre(dim1, dim2):
    A = np.random.normal(0, 1, (dim1, dim2))
    B = np.random.normal(0, 1, (dim1, dim2))
    X = A + 1j*B
    return X, A, B


def generate_state(d, rank):
    X, _, _ = generate_ginibre(d, rank)

    state = X@X.conj().T/np.trace(X@X.conj().T)
    return state


def generate_unitary(X):
    Q, R = np.linalg.qr(X)
    R = np.diag(R)
    sign = R/np.abs(R)
    U = Q@np.diag(sign)

    return U


def reshuffle_choi(choi):
    d = int(np.sqrt(choi.shape[0]))
    choi = choi.reshape(d,d,d,d).swapaxes(1,2).reshape(d**2, d**2)
    return choi


def choi_spectrum(choi):
    choi = reshuffle_choi(choi)
    eig, _ = np.linalg.eig(choi)

    x = np.real(eig)
    y = np.imag(eig)

    return np.array([x, y])


def choi_steady_state(choi):
    d = int(np.sqrt(choi.shape[0]))
    choi = reshuffle_choi(choi)
    _, eig_vec = np.linalg.eig(choi)

    steady_state = eig_vec[:,0]
    steady_state = steady_state.reshape(d, d)

    return steady_state
