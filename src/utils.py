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

def train(num_iter):
    def _train(model):
        model.train(num_iter=num_iter,
                    use_adam = True,
                    verbose = False)
        return model

    return _train


def train_parallel(num_iter, model_list, num_workers):
    with mp.Pool(num_workers) as p:
        p.map(train(num_iter), model_list)


def generate_bitstring_circuits(n):
    circuit_list = []
    for i in range(2**n):
        q_reg = qk.QuantumRegister(n)
        c_reg = qk.ClassicalRegister(n)
        circuit = qk.QuantumCircuit(q_reg, c_reg)
        config = numberToBase(i, 2, n)
        for j, index in enumerate(config):
            if index:
                circuit.x(j)
        circuit.measure(q_reg, c_reg)
        circuit_list.append(circuit)

    return circuit_list


def generate_corruption_matrix(n, counts_list):
    corr_mat = np.zeros((2**n, 2**n))
    for i, counts in enumerate(counts_list):
        for string, value in counts.items():
            index = int(string[::-1], 2)
            corr_mat[i, index] = value

    corr_mat = corr_mat/sum(counts_list[0].values())
    return corr_mat


def counts_to_vector(counts, n):
    vec = np.zeros(2**n)
    for string, value in counts.items():
        index = int(string, 2)
        vec[index] = value
    vec = vec/sum(counts.values())
    return vec

def vector_to_counts(vector, n):
    counts = {}
    for i in range(2**n):
        config = reversed(numberToBase(i, 2, n))
        string = "".join([str(index) for index in config])

        counts[string] = vector[i]

    return counts


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
