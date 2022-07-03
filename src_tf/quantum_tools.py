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



def partial_trace(X, discard_first = True):
    d = int(np.sqrt(X.shape[0]))
    X = tf.reshape(X, (d,d,d,d))
    if discard_first:
        Y = tf.einsum("ijik->jk", X)
    else:
        Y = tf.einsum("jiki->jk", X)
    return Y


def state_fidelity(A, B):

    sqrtB = sqrtm(B)
    C = sqrtB@A@sqrtB

    fidelity = tf.trace(sqrtm(C))
    return tf.abs(fidelity)**2


def channel_fidelity(map_A, map_B):
    choi_A = maps_to_choi([map_A])
    choi_B = maps_to_choi([map_B])
    d_squared = choi_A.shape[0]
    fidelity = state_fidelity(choi_A, choi_B)/d_squared

    return fidelity


def expectation_value(probs, observable):
    ev = tf.abs(tf.reduce_sum(probs*observable, axis = 1))
    return ev


#@profile
def generate_ginibre(dim1, dim2, trainable = False):
    A = tf.cast(tf.random.normal((dim1, dim2), 0, 1), dtype = tf.complex64)
    B = tf.cast(tf.random.normal((dim1, dim2), 0, 1), dtype = tf.complex64)
    if trainable:
        A = tf.Variable(A, trainable = True)
        B = tf.Variable(B, trainable = True)

    X = A + 1j*B
    return X, A, B


def generate_state(d, rank):
    X, _, _ = generate_ginibre(d, rank)

    state = X@X.conj().T/np.trace(X@X.conj().T)
    return state


def generate_unitary(G):
    Q, R = tf.linalg.qr(G, full_matrices = False)
    D = tf.linalg.tensor_diag_part(R)
    D = tf.math.sign(D)
    D = tf.linalg.diag(D)
    U = Q@D

    return U
