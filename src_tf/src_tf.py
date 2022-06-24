import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import tensorflow as tf

from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm



def generate_ginibre(dim1, dim2, real = False):
    ginibre = np.random.normal(0, 1, (dim1, dim2))
    if not real:
         ginibre = ginibre + 1j*np.random.normal(0, 1, (dim1, dim2))
    return tf.convert_to_tensor(ginibre, dtype=tf.complex128)


def generate_state(dim1, dim2):
    X = generate_ginibre(dim1, dim2)

    state = X@X.conj().T/torch.trace(X@X.conj().T)
    return state
