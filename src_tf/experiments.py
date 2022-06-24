import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm
from quantum_tools import *
from utils import *

#@profile
def prepare_input(config, return_mode = "density"):
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

    if return_mode == "density":
        result = tf.convert_to_tensor(DensityMatrix(circuit).data, dtype=tf.complex128)
    if return_mode == "unitary":
        result = Operator(circuit).data
    if return_mode == "circuit":
        result = circuit

    return result


def pauli_observable(config, trace = True, return_mode = "density"):

    X = tf.Tensor([[0, 1], [1, 0]])
    Y = tf.Tensor([[0, -1j], [1j, 0]])
    Z = tf.Tensor([[1, 0], [0, -1]])
    I = tf.eye(2)

    basis = [X, Y, Z]

    if trace:
        basis.append(I)

    if return_mode == "density":
        string = [basis[idx] for idx in config]
        result = tf.convert_to_tensor(kron(*string), dtype=tf.complex128)

    if return_mode == "circuit":

        n_sub = sum([idx != 3 for idx in config])
        n_count = 0

        q_reg = qk.QuantumRegister(len(config))
        c_reg = qk.ClassicalRegister(n_sub)
        circuit = qk.QuantumCircuit(q_reg, c_reg)

        for i, index in enumerate(reversed(config)):
            if index == 0:
                circuit.h(i)

            if index == 1:
                circuit.sdg(i)
                circuit.h(i)

            if index == 2:
                pass    #measure in computational basis

            if index != 3:
                circuit.measure(q_reg[i], c_reg[n_count])
                n_count += 1

        result = circuit

    return result


def generate_pauli_circuits(circuit_target, N, trace=True):
    n = len(circuit_target.qregs[0])
    state_index, observ_index = index_generator(n, N, trace)

    if trace:
        num_observ = 4
    else:
        num_observ = 3

    input_list = []
    circuit_list = []
    for i, j in zip(state_index, observ_index):

        config = numberToBase(i, 6, n)
        state = prepare_input(config)
        state_circuit = prepare_input(config, return_mode = "circuit")

        config = numberToBase(j, num_observ, n)
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


def expected_parity(counts):
    shots = sum(counts.values())
    parity = 0
    for string, count in counts.items():

        if string.count("1")%2 == 0:
            parity += count
        else:
            parity -= count

    parity = parity/shots
    return parity


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


def generate_corruption_matrix(counts_list):
    n = len(counts_list[0].keys()[0])
    corr_mat = np.zeros((2**n, 2**n))
    for i, counts in enumerate(counts_list):
        for string, value in counts.items():
            index = int(string[::-1], 2)
            corr_mat[i, index] = value

    corr_mat = corr_mat/sum(counts_list[0].values())
    return corr_mat


def counts_to_vector(counts):
    n = len(counts.keys()[0])
    vec = np.zeros(2**n)
    for string, value in counts.items():
        index = int(string, 2)
        vec[index] = value
    vec = vec/sum(counts.values())
    return vec


def vector_to_counts(vector):
    n = int(np.log2(len(vector)))
    counts = {}
    for i in range(2**n):
        config = reversed(numberToBase(i, 2, n))
        string = "".join([str(index) for index in config])

        counts[string] = vector[i]

    return counts
