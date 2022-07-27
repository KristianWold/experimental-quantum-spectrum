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


def generate_pauli_circuits(n = None, 
                            circuit_target = None, 
                            N = None, 
                            trace=False):
    state_index, observ_index = index_generator(n, N, trace=trace)

    if trace:
        num_observ = 4
    else:
        num_observ = 3

    input_list = [[],[]]
    circuit_list = []
    for i, j in zip(state_index, observ_index):

        config1 = numberToBase(i, 6, n)
        state = prepare_input(config1, return_mode = "density")


        config2 = numberToBase(j, num_observ, n)
        U_basis, _ = pauli_observable(config2, return_mode = "unitary")


        input_list[0].append(state)
        input_list[1].append(U_basis)

        if circuit_target is not None:
            state_circuit = prepare_input(config1, return_mode = "circuit")
            observable_circuit = pauli_observable(config2, return_mode = "circuit")

            circuit = state_circuit
            circuit.barrier()
            circuit = circuit.compose(circuit_target)
            circuit.barrier()
            circuit.add_register(observable_circuit.cregs[0])
            circuit = circuit.compose(observable_circuit)

            circuit_list.append(circuit)

    input_list[0] = tf.convert_to_tensor(input_list[0], dtype=tf.complex64)
    input_list[1] = tf.convert_to_tensor(input_list[1], dtype=tf.complex64)

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
        for string, value in counts.items():
            index = int(string, 2)
            corr_mat[index, i] = value

    corr_mat = corr_mat/sum(counts_list[0].values())
    return corr_mat


def counts_to_probs(counts_list):
    N = len(counts_list)
    n = len(list(counts_list[0].keys())[0])
    probs = np.zeros((N, 2**n))
    for i in range(N):
        for string, value in counts_list[i].items():
            index = int(string, 2)
            probs[i, index] = value
    probs = probs/sum(counts_list[0].values())
    probs = tf.convert_to_tensor(probs, dtype=tf.complex64)
    return probs


def corr_mat_to_povm(corr_mat):
    d = corr_mat.shape[0]
    povm = []
    for i in range(d):
        M = tf.cast(np.diag(corr_mat[i,:]), dtype=tf.complex64)
        povm.append(M)

    povm = tf.convert_to_tensor(povm, dtype=tf.complex64)

    return povm


def povm_ideal(n):
    povm = corr_mat_to_povm(np.eye(2**n))
    return povm


def measurement(state, U_basis=None, povm=None):
    d = state.shape[1]
    if U_basis is None:
        U_basis = tf.eye(d, dtype=tf.complex64)
        U_basis = tf.expand_dims(U_basis, axis = 0)

    if povm is None:
        povm = corr_mat_to_povm(np.eye(d))

    Ustate = tf.matmul(U_basis, state)
    UstateU = tf.matmul(Ustate, U_basis, adjoint_b=True)

    state = tf.expand_dims(UstateU, axis=1)
    povm = tf.expand_dims(povm, axis=0)

    probs = tf.linalg.trace(state@povm)

    return probs


def parity_observable(n, trace_index_list=[]):
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    observable = n*[Z]
    for index in trace_index_list:
        observable[index] = I

    observable = np.diag(kron(*observable))
    return observable
