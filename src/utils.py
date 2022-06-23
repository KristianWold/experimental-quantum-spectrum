import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm



def generate_pauli_circuits(circuit_target, N):
    n = len(circuit_target.qregs[0])
    state_index, observ_index = index_generator(n, N)

    input_list = []
    circuit_list = []
    for i, j in zip(state_index, observ_index):

        config = numberToBase(i, 6, n)
        state = prepare_input(config)
        state_circuit = prepare_input(config, return_mode = "circuit")

        config = numberToBase(j, 4, n)
        observable = pauli_observable(config)
        observable_circuit = pauli_observable(config, return_mode = "circuit")

        input_list.append([state, observable])
        circuit = state_circuit
        circuit.barrier()
        circuit = circuit.compose(circuit_target)
        circuit.barrier()
        circuit.add_register(observable_circuit.cregs[0])
        circuit = circuit.compose(observable_circuit)

        circuit_list.append(circuit)

    return input_list, circuit_list


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


def index_generator(n, N=None):
    index_list1 = np.arange(0, 6**n)
    index_list2 = np.arange(1, 4**n)
    if N is None:
        N = len(index_list1)*len(index_list2)

    index_list1, index_list2 = np.meshgrid(index_list1, index_list2)
    index_list = np.vstack([index_list1.flatten(), index_list2.flatten()]).T
    np.random.shuffle(index_list)

    return index_list[:N, 0], index_list[:N, 1]
