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
            pass
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
        state = DensityMatrix(circuit.reverse_bits()).data
    if return_mode == "unitary":
        state = Operator(circuit.reverse_bits()).data
    if return_mode == "circuit":
        state = circuit.reverse_bits()

    return state


def pauli_observable(config, return_mode = "density"):

    n = len(config)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    basis = [X, Y, Z, I]

    if return_mode == "density":
        string = [basis[idx] for idx in config]
        result = kron(*string)


    q_reg = qk.QuantumRegister(n)
    c_reg = qk.ClassicalRegister(n)
    circuit = qk.QuantumCircuit(q_reg, c_reg)

    for i, index in enumerate(config):
        if index == 0:
            circuit.h(i)

        if index == 1:
            circuit.sdg(i)
            circuit.h(i)

        if index == 2:
            pass    #measure in computational basis


    if return_mode == "circuit":
        circuit.measure(q_reg, c_reg)
        result = circuit.reverse_bits()

    if return_mode == "unitary":
        trace_index_list = []

        for i, idx in enumerate(config):
            if idx == 3:
                trace_index_list.append(i)

        observable = parity_observable(n, trace_index_list)

        result = [Operator(circuit.reverse_bits()).data, observable]

    return result


def generate_pauli_circuits(circuit_target, N, trace=False):
    n = len(circuit_target.qregs[0])
    state_index, observ_index = index_generator(n, N, trace=trace)

    if trace:
        num_observ = 4
    else:
        num_observ = 3

    input_list = []
    circuit_list = []
    for i, j in zip(state_index, observ_index):

        config = numberToBase(i, 6, n)
        state = prepare_input(config, return_mode = "density")
        state_circuit = prepare_input(config, return_mode = "circuit")

        config = numberToBase(j, num_observ, n)
        U_basis, observable = pauli_observable(config, return_mode = "unitary")
        observable_circuit = pauli_observable(config, return_mode = "circuit")

        input_list.append([state, U_basis, observable])
        circuit = state_circuit
        circuit.barrier()
        circuit = circuit.compose(circuit_target)
        circuit.barrier()
        circuit.add_register(observable_circuit.cregs[0])
        circuit = circuit.compose(observable_circuit)

        circuit_list.append(circuit)

    return input_list, circuit_list


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
        circuit_list.append(circuit.reverse_bits())

    return circuit_list


def generate_corruption_matrix(counts_list):
    n = len(list(counts_list[0].keys())[0])
    corr_mat = np.zeros((2**n, 2**n))
    for i, counts in enumerate(counts_list):
        #idx = numberToBase(i, 2, n)
        #idx = int("".join([str(j) for j in idx])[::-1], 2)
        for string, value in counts.items():
            index = int(string, 2)
            corr_mat[index, i] = value

    corr_mat = corr_mat/sum(counts_list[0].values())
    return corr_mat


def counts_to_probs(counts):
    n = len(list(counts.keys())[0])
    probs = np.zeros(2**n)
    for string, value in counts.items():
        index = int(string, 2)
        probs[index] = value
    probs = probs/sum(counts.values())
    return probs


def probs_to_counts(probs):
    n = int(np.log2(len(probs)))
    counts = {}
    for i in range(2**n):
        config = numberToBase(i, 2, n)
        string = "".join([str(index) for index in config])

        counts[string] = probs[i]

    return counts


def corr_mat_to_povm(corr_mat):
    d = corr_mat.shape[0]
    povm = []
    for i in range(d):
        M = tf.cast(np.diag(corr_mat[i,:]), dtype=tf.complex64)
        povm.append(M)

    return povm


def povm_ideal(n):
    povm = corr_mat_to_povm(np.eye(2**n))
    return povm


def expectation_value(probs, observable):
    ev = np.abs(np.sum(probs*observable))
    return ev


def measurement(state, U_basis=None, povm=None):
    d = state.shape[0]
    if U_basis is None:
        U_basis = tf.eye(d, dtype=tf.complex64)

    if povm is None:
        povm = corr_mat_to_povm(np.eye(d))

    state = U_basis@state@tf.linalg.adjoint(U_basis)
    probs = []
    for i, M in enumerate(povm):
        probs.append(tf.linalg.trace(state@M))
    probs = tf.convert_to_tensor(probs)

    return probs


def parity_observable(n, trace_index_list=[]):
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    observable = n*[Z]
    for index in trace_index_list:
        observable[index] = I

    observable = np.diag(kron(*observable))
    return observable
