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
from set_precision import *

# @profile
def prepare_input(config, return_mode="density"):
    """0 = |0>, 1 = |1>, 2 = |+>, 3 = |->, 4 = |i+>, 5 = |i+>"""
    n = len(config)
    circuit = qk.QuantumCircuit(n)
    for i, gate in enumerate(config):
        if gate == 0:
            pass
        if gate == 1:
            circuit.rx(np.pi, i)
        if gate == 2:
            circuit.ry(np.pi / 2, i)
        if gate == 3:
            circuit.ry(-np.pi / 2, i)
        if gate == 4:
            circuit.rx(-np.pi / 2, i)
        if gate == 5:
            circuit.rx(np.pi / 2, i)

    if return_mode == "density":
        state = DensityMatrix(circuit.reverse_bits()).data
    if return_mode == "unitary":
        state = Operator(circuit.reverse_bits()).data
    if return_mode == "circuit":
        state = circuit.reverse_bits()

    if return_mode == "circuit_measure":
        circuit.add_register(qk.ClassicalRegister(n))
        circuit.measure(circuit.qregs[0], circuit.cregs[0])
        state = circuit.reverse_bits()

    return state


def prepare_input_entangled(config, return_mode="density"):
    """0 = |0>, 1 = |1>, 2 = |+>, 3 = |->, 4 = |i+>, 5 = |i+>"""
    n = len(config)-1
    circuit = qk.QuantumCircuit(n)
    for i, gate in enumerate(config[:-1]):
        if gate == 0:
            pass
        if gate == 1:
            circuit.rx(np.pi, i)
        if gate == 2:
            circuit.ry(np.pi / 2, i)
        if gate == 3:
            circuit.ry(-np.pi / 2, i)
        if gate == 4:
            circuit.rx(-np.pi / 2, i)
        if gate == 5:
            circuit.rx(np.pi / 2, i)
    
    cnot_position = config[-1]
    if cnot_position//(n-1) != 0:
        cnot_position = cnot_position%(n-1)
        circuit.cx(cnot_position+1, cnot_position)
    else:
        circuit.cx(cnot_position, cnot_position+1)

    if return_mode == "density":
        state = DensityMatrix(circuit.reverse_bits()).data
    if return_mode == "unitary":
        state = Operator(circuit.reverse_bits()).data
    if return_mode == "circuit":
        state = circuit.reverse_bits()

    if return_mode == "circuit_measure":
        circuit.add_register(qk.ClassicalRegister(n))
        circuit.measure(circuit.qregs[0], circuit.cregs[0])
        state = circuit.reverse_bits()

    return state


def pauli_observable(config, return_mode="density"):

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
            circuit.ry(-np.pi / 2, i)

        if index == 1:
            circuit.rx(np.pi / 2, i)

        if index == 2:
            pass  # measure in computational basis


    

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


def pauli_observable_entangled(config, return_mode="density"):

    n = len(config)-1
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

    cnot_position = config[-1]
    if cnot_position//(n-1) == 0:
        circuit.cx(cnot_position, cnot_position+1)
    else:
        cnot_position = cnot_position%(n-1)
        circuit.cx(cnot_position+1, cnot_position)

    for i, index in enumerate(config[:-1]):
        if index == 0:
            circuit.ry(-np.pi / 2, i)

        if index == 1:
            circuit.rx(np.pi / 2, i)

        if index == 2:
            pass  # measure in computational basis

    if return_mode == "circuit":
        circuit.measure(q_reg, c_reg)
        result = circuit.reverse_bits()

    if return_mode == "unitary":
        trace_index_list = []

        for i, idx in enumerate(config):
            if idx == 3:
                trace_index_list.append(i)

        observable = None

        result = [Operator(circuit.reverse_bits()).data, observable]

    return result



def generate_pauli_circuits(n=None, circuit_target=None, N=None, trace=False):
    state_index, observ_index = index_generator(n, N, trace=trace)

    if trace:
        num_observ = 4
    else:
        num_observ = 3

    input_list = [[], []]
    circuit_list = []
    for i, j in zip(state_index, observ_index):

        config1 = numberToBase(i, 6, n)
        U_prep = prepare_input(config1, return_mode="unitary")

        config2 = numberToBase(j, num_observ, n)
        U_basis, _ = pauli_observable(config2, return_mode="unitary")

        input_list[0].append(U_prep)
        input_list[1].append(U_basis)

        if circuit_target is not None:
            state_circuit = prepare_input(config1, return_mode="circuit")
            observable_circuit = pauli_observable(config2, return_mode="circuit")

            circuit = state_circuit
            circuit.barrier()
            circuit = circuit.compose(circuit_target)
            circuit.barrier()
            circuit.add_register(observable_circuit.cregs[0])
            circuit = circuit.compose(observable_circuit)

            circuit_list.append(circuit)

    input_list[0] = tf.convert_to_tensor(input_list[0], dtype=precision)
    input_list[1] = tf.convert_to_tensor(input_list[1], dtype=precision)

    return input_list, circuit_list


def generate_pauli_circuits_entangled(n=None, circuit_target=None, N=None, trace=False):
    state_index, observ_index = index_generator(n, N, trace=trace)

    if trace:
        num_observ = 4
    else:
        num_observ = 3

    input_list = [[], []]
    circuit_list = []
    for i, j in zip(state_index, observ_index):

        config1 = numberToBase(i, 6, n)
        config1.append(np.random.randint(0, 2*(n-1)))
        U_prep = prepare_input_entangled(config1, return_mode="unitary")

        config2 = numberToBase(j, num_observ, n)
        config2.append(np.random.randint(0, 2*(n-1)))
        U_basis, _ = pauli_observable_entangled(config2, return_mode="unitary")

        input_list[0].append(U_prep)
        input_list[1].append(U_basis)

        if circuit_target is not None:
            state_circuit = prepare_input_entangled(config1, return_mode="circuit")
            observable_circuit = pauli_observable_entangled(config2, return_mode="circuit")

            circuit = state_circuit
            circuit.barrier()
            circuit = circuit.compose(circuit_target)
            circuit.barrier()
            circuit.add_register(observable_circuit.cregs[0])
            circuit = circuit.compose(observable_circuit)

            circuit_list.append(circuit)

    input_list[0] = tf.convert_to_tensor(input_list[0], dtype=precision)
    input_list[1] = tf.convert_to_tensor(input_list[1], dtype=precision)

    return input_list, circuit_list


def generate_pauliInput_circuits(n=None):
    input_list = []
    circuit_list = []
    for i in range(6**n):

        config = numberToBase(i, 6, n)
        U_prep = prepare_input(config, return_mode="unitary")
        circuit = prepare_input(config, return_mode="circuit_measure")

        input_list.append(U_prep)
        circuit_list.append(circuit)

    input_list = tf.convert_to_tensor(input_list, dtype=precision)

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


def counts_to_probs(counts_list):
    N = len(counts_list)
    n = len(list(counts_list[0].keys())[0])
    probs = np.zeros((N, 2**n))
    for i in range(N):
        for string, value in counts_list[i].items():
            index = int(string, 2)
            probs[i, index] = value
    probs = probs / sum(counts_list[0].values())
    probs = tf.convert_to_tensor(probs, dtype=precision)
    return probs


def pqc_basic(n, L):
    theta_list = [np.random.uniform(-np.pi, np.pi, 2 * n) for i in range(L)]
    circuit = qk.QuantumCircuit(n)
    for theta in theta_list:
        for i in range(n):
            circuit.ry(theta[i], i)
            circuit.rz(theta[i + n], i)

        for i in range(n - 1):
            circuit.cx(i, i + 1)

    return circuit


def pqc_expressive(n, L):
    theta_list = [np.random.uniform(0, 2 * np.pi, 4 * n) for i in range(L)]
    circuit = qk.QuantumCircuit(n)
    for theta in theta_list:
        for i in range(n):
            circuit.ry(theta[i], i)

        for i in range(n):
            circuit.crx(theta[i + n], i, (i + 1) % n)

        for i in range(n):
            circuit.ry(theta[i + 2 * n], i)

        for i in reversed(list(range(n))):
            circuit.crx(theta[3 * n + i], (i + 1) % n, i)

    return circuit


def pqc_more_expressive(n, L):
    theta_list = [np.random.uniform(-np.pi, np.pi, 4 * n) for i in range(L)]
    circuit = qk.QuantumCircuit(n)
    for theta in theta_list:
        for i in range(n):
            circuit.ry(theta[i], i)
            circuit.rz(theta[i + n], i)

        for i in range(n):
            circuit.cx(i, (i + 1) % n)

        for i in range(n):
            circuit.ry(theta[i + 2 * n], i)
            circuit.rx(theta[i + 3 * n], i)

        for i in range(n):
            circuit.cx(n - i - 1, n - (i + 1) % n - 1)

    return circuit


def parity_observable(n, trace_index_list=[]):
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    observable = n * [Z]
    for index in trace_index_list:
        observable[index] = I

    observable = np.diag(kron(*observable))
    return observable
