import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm
import multiprocessing as mp

def numberToBase(n, b, num_digits):
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b

    while len(digits)<num_digits:
        digits.append(0)
    return digits[::-1]


def kron(*args):
    length = len(args)
    A = args[0]
    for i in range(1, length):
        A = np.kron(A, args[i])

    return A


def index_generator(n, N=None, trace=True):

    index_list1 = np.arange(0, 6**n)
    if trace:
        index_list2 = np.arange(0, 4**n-1)
    else:
        index_list2 = np.arange(0, 3**n)

    if N is None:
        N = len(index_list1)*len(index_list2)

    index_list1, index_list2 = np.meshgrid(index_list1, index_list2)
    index_list = np.vstack([index_list1.flatten(), index_list2.flatten()]).T
    np.random.shuffle(index_list)

    return index_list[:N, 0], index_list[:N, 1]
