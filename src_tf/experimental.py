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
        result = Operator(circuit.reverse_bits()).data

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
        U_basis = pauli_observable(config2, return_mode="unitary")

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


class ExecuteAndCollect:

    def setup_circuits(self, circuit_target_list=None, N_map=None, N_spam=None):
        self.circuit_target_list = circuit_target_list

        self.n = len(circuit_target_list[0].qregs[0])
        self.data_list = []

        for circuit_target in self.circuit_target_list:
            inputs_map, circuit_list_map = generate_pauli_circuits(self.n, circuit_target, N = N_map)
            inputs_spam, circuit_list_spam = generate_pauliInput_circuits(self.n)

            self.data_list.append([inputs_map, circuit_list_map, inputs_spam, circuit_list_spam])

    def execute_circuits(self, backend, shots_map, shots_spam, filename=None, concatenate=False):
        self.result_list = []
        self.shots_map = shots_map
        self.shots_spam = shots_spam

        self.result_list = []

        for i, data in enumerate(self.data_list):
            inputs_map, circuit_list_map, inputs_spam, circuit_list_spam = data
            if concatenate:
                circuit_list = circuit_list_map + circuit_list_spam
                counts_list = self.runner(circuit_list, backend, shots=shots_map)
                counts_map = counts_list[:len(circuit_list_map)]
                counts_spam = counts_list[len(circuit_list_map):]
            else:
                counts_map = self.runner(circuit_list_map, backend, shots=shots_map)
                counts_spam = self.runner(circuit_list_spam, backend, shots=shots_spam)
            
            probs_map = counts_to_probs(counts_map)
            probs_spam = counts_to_probs(counts_spam)
            self.result_list.append([inputs_map, probs_map, inputs_spam, probs_spam])

            with open("../../data/" + filename + str(i), 'wb') as handle:
                pickle.dump(self.result_list[-1], handle)

    def runner(self, circuit_list, backend, shots, scheduling_method = "asap"):
        N = len(circuit_list)
        num_batches = (N+500-1)//500
        circuit_batch_list = [circuit_list[500*i: 500*(i+1)] for i in range(num_batches)]
        counts_list = []
        for i, circuit_batch in enumerate(tqdm(circuit_batch_list)):
            num_parcels = (len(circuit_batch) + 100 - 1)//100
            circuit_parcel_list = [circuit_batch[100*j: 100*(j+1)] for j in range(num_parcels)]
            job_list = []
            
            for circuit_parcel in circuit_parcel_list:
                trans_circ_list = qk.transpile(circuit_parcel, 
                                            backend, 
                                            optimization_level = 0, 
                                            seed_transpiler=42, 
                                            scheduling_method = scheduling_method)
                
                job = backend.run(trans_circ_list, shots = shots)
                
                job_list.append(job)
            
            result_list = []
            for job in tqdm(job_list):
                result_list.append(job.result())
                
            for result, circuit_parcel in zip(result_list, circuit_parcel_list):
                counts_list.extend([result.get_counts(circuit) for circuit in circuit_parcel]) 
        
        return counts_list  

