import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
from qiskit.quantum_info import DensityMatrix, Operator, random_unitary
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm

from quantum_tools import *
from quantum_channel import *
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


def generate_sandwich_circuits(target_circuit, input_circuit_list, output_circuit_list):
    circuit_list = []
    n = len(input_circuit_list[0].qregs[0])
    for i in range(len(input_circuit_list)):
        circuit = input_circuit_list[i]
        circuit.barrier()
        circuit = circuit.compose(target_circuit)
        circuit.barrier()
        circuit = circuit.compose(output_circuit_list[i])

        circuit.add_register(qk.ClassicalRegister(n))
        circuit.measure(circuit.qregs[0], circuit.cregs[0])

        circuit_list.append(circuit)

    return circuit_list


class SphereStrings:
    def __init__(self, n, N):
        self.n = n
        self.d = 2**n
        self.N = N

        self.I = tf.convert_to_tensor([[1, 0], [0, 1]], dtype=precision)
        self.I = tf.repeat(self.I[None, :, :], self.n, axis=0)
        self.I = tf.repeat(self.I[None, :, :, :], self.N, axis=0)

        self.X = tf.convert_to_tensor([[0, 1], [1, 0]], dtype=precision)
        self.X = tf.repeat(self.X[None, :, :], self.n, axis=0)
        self.X = tf.repeat(self.X[None, :, :, :], self.N, axis=0)

        self.Y = tf.convert_to_tensor([[0, -1j], [1j, 0]], dtype=precision)
        self.Y = tf.repeat(self.Y[None, :, :], self.n, axis=0)
        self.Y = tf.repeat(self.Y[None, :, :, :], self.N, axis=0)

        # self.parameters = tf.random.normal((self.N, 2 * self.n, 1, 1), 0, 1)
        self.parameters = tf.random.uniform((self.N, 2 * self.n, 1, 1), -np.pi, np.pi)
        self.parameters = tf.Variable(self.parameters, trainable=True)
        self.parameter_list = [self.parameters]

    def generate(self):
        self.angles = tf.cast(self.parameters, dtype=precision)
        # self.angles = tf.cast(2*np.pi*tf.math.tanh(self.parameters), dtype = precision)
        rx = (
            tf.math.cos(self.angles[:, 0 : self.n] / 2) * self.I
            - 1j * tf.math.sin(self.angles[:, 0 : self.n] / 2) * self.X
        )
        ry = (
            tf.math.cos(self.angles[:, self.n :] / 2) * self.I
            - 1j * tf.math.sin(self.angles[:, self.n :] / 2) * self.Y
        )
        self.strings = ry @ rx

    def fidelity(self):
        self.generate()
        # fid = 0
        # for i in range(self.N):
        #    for j in range(self.N):
        #        A = self.strings[i]@tf.linalg.adjoint(self.strings[j])
        #        A = tf.linalg.trace(A)
        #        A = tf.math.reduce_prod(A)
        #        fid += tf.abs(A)**2

        # fid = (fid/self.N**2 + self.d)/(self.d**2 + self.d)
        # A = tf.tensordot(self.string, tf.linalg.adjoint(self.string), axes = 0)

        A = tf.linalg.einsum(
            "a...ij,b...jk -> ab...ik", self.strings, tf.linalg.adjoint(self.strings)
        )
        A = tf.linalg.trace(A)
        A = tf.abs(tf.math.reduce_prod(A, axis=2)) ** 2

        fid = (A + self.d) / (self.d**2 + self.d)

        fid = tf.linalg.set_diag(fid, tf.zeros_like(fid[:, 0]))
        fid = tf.math.reduce_max(fid, axis=1)
        fid = tf.math.reduce_sum(fid) / self.N

        """
        A = tf.linalg.einsum('abij,cdjk -> abcdik', self.strings, tf.linalg.adjoint(self.strings))
        A = tf.linalg.trace(A)
        fid = (tf.math.reduce_sum(tf.abs(A)**2)/self.N**2 + self.d)/(self.d**2 + self.d)
        """
        return fid

    def generate_circuits(self, grid=False):
        circuit_list = []
        unitary_list = []
        self.generate()
        angles = np.real(self.angles.numpy()[:, :, 0, 0])
        for i in range(self.N):
            q_reg = qk.QuantumRegister(self.n)
            circuit = qk.QuantumCircuit(q_reg)
            for j in range(self.n):
                circuit.rx(angles[i, j], j)
                circuit.ry(angles[i, j + self.n], j)
            circuit_list.append(circuit)
            unitary_list.append(Operator(circuit.reverse_bits()).data)

        unitary_list = tf.convert_to_tensor(unitary_list, dtype=precision)
        return circuit_list, unitary_list

    def optimize(self, steps):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        for i in tqdm(range(steps)):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.parameters)

                loss = self.fidelity()
                grads = tape.gradient(loss, self.parameter_list)
                optimizer.apply_gradients(zip(grads, self.parameter_list))
            # print(loss)


class GridStrings:
    def __init__(self, n, N, grid_points=10):
        self.n = n
        self.d = 2**n
        self.N = N

        angle_list1 = np.linspace(-np.pi, np.pi, grid_points**n)
        index_list2 = np.arange(-np.pi, np.pi, grid_points**n)

        index_list1, index_list2 = np.meshgrid(index_list1, index_list2)

        N_ = np.ceil(np.sqrt(N)).astype(int)
        angle_linspace = np.linspace(-np.pi, np.pi, N_, endpoint=False)
        angle1, angle2 = np.meshgrid(angle_linspace, angle_linspace)
        grid_angles = np.stack([angle1.flatten(), angle2.flatten()], axis=1)
        np.random.shuffle(grid_angles)
        self.grid_angles = grid_angles[:N]

    def generate_circuits(self, grid=False):
        circuit_list = []
        unitary_list = []
        for i in range(self.N):
            q_reg = qk.QuantumRegister(self.n)
            circuit = qk.QuantumCircuit(q_reg)
            for j in range(self.n):
                circuit.rx(self.grid_angles[i, j], j)
                circuit.ry(self.grid_angles[i, j + self.n], j)
            circuit_list.append(circuit)
            unitary_list.append(Operator(circuit.reverse_bits()).data)

        unitary_list = tf.convert_to_tensor(unitary_list, dtype=precision)
        return circuit_list, unitary_list


class HaarStrings:
    def __init__(self, n, N, seed=42):
        self.n = n
        self.d = 2**n
        self.N = N
        self.RNG = np.random.default_rng(seed=seed)

    def generate(self):
        self.strings = []
        for i in range(self.N):
            U = [
                tf.cast(random_unitary(2, seed=self.RNG).data, dtype=precision)
                for j in range(self.n)
            ]
            self.strings.append(U)

    def fidelity(self):
        self.generate()

        strings_tensor = tf.cast(self.strings, dtype=precision)

        A = tf.linalg.einsum(
            "a...ij,b...jk -> ab...ik",
            strings_tensor,
            tf.linalg.adjoint(strings_tensor),
        )
        A = tf.linalg.trace(A)
        A = tf.abs(tf.math.reduce_prod(A, axis=2)) ** 2

        fid = (A + self.d) / (self.d**2 + self.d)

        fid = tf.linalg.set_diag(fid, tf.zeros_like(fid[:, 0]))
        fid = tf.math.reduce_max(fid, axis=1)
        fid = tf.math.reduce_sum(fid) / self.N

        return fid

    def generate_circuits(self):
        unitary_list = []
        self.generate()
        for U in self.strings:
            unitary_list.append(kron(*U))
        unitary_list = tf.convert_to_tensor(unitary_list, dtype=precision)
        return None, unitary_list


class HaarInput:
    def __init__(self, n, N):
        self.n = n
        self.d = 2**n
        self.N = N

    def generate(self):
        self.strings = []
        for i in range(self.N):
            seed = np.random.randint(0, 10**6)
            U = tf.cast(random_unitary(self.d, seed=seed).data, dtype=precision)
            self.strings.append(U)

        self.strings = tf.cast(self.strings, dtype=precision)

    def fidelity(self):
        self.generate()

        strings_tensor = tf.cast(self.strings, dtype=precision)

        A = tf.linalg.einsum(
            "a...ij,b...jk -> ab...ik",
            strings_tensor,
            tf.linalg.adjoint(strings_tensor),
        )
        A = tf.linalg.trace(A)
        A = tf.abs(tf.math.reduce_prod(A, axis=2)) ** 2

        fid = (A + self.d) / (self.d**2 + self.d)

        fid = tf.linalg.set_diag(fid, tf.zeros_like(fid[:, 0]))
        fid = tf.math.reduce_max(fid, axis=1)
        fid = tf.math.reduce_sum(fid) / self.N

        return fid

    def generate_circuits(self):
        unitary_list = []
        self.generate()
        unitary_list = self.strings
        return None, unitary_list


class SphereStrings:
    def __init__(self, n, N):
        self.n = n
        self.d = 2**n
        self.N = N

        self.I = tf.convert_to_tensor([[1, 0], [0, 1]], dtype=precision)
        self.I = tf.repeat(self.I[None, :, :], self.n, axis=0)
        self.I = tf.repeat(self.I[None, :, :, :], self.N, axis=0)

        self.X = tf.convert_to_tensor([[0, 1], [1, 0]], dtype=precision)
        self.X = tf.repeat(self.X[None, :, :], self.n, axis=0)
        self.X = tf.repeat(self.X[None, :, :, :], self.N, axis=0)

        self.Y = tf.convert_to_tensor([[0, -1j], [1j, 0]], dtype=precision)
        self.Y = tf.repeat(self.Y[None, :, :], self.n, axis=0)
        self.Y = tf.repeat(self.Y[None, :, :, :], self.N, axis=0)

        # self.parameters = tf.random.normal((self.N, 2 * self.n, 1, 1), 0, 1)
        self.parameters = tf.random.uniform((self.N, 2 * self.n, 1, 1), -np.pi, np.pi)
        self.parameters = tf.Variable(self.parameters, trainable=True)
        self.parameter_list = [self.parameters]

    def generate(self):
        self.angles = tf.cast(self.parameters, dtype=precision)
        # self.angles = tf.cast(2*np.pi*tf.math.tanh(self.parameters), dtype = precision)
        rx = (
            tf.math.cos(self.angles[:, 0 : self.n] / 2) * self.I
            - 1j * tf.math.sin(self.angles[:, 0 : self.n] / 2) * self.X
        )
        ry = (
            tf.math.cos(self.angles[:, self.n :] / 2) * self.I
            - 1j * tf.math.sin(self.angles[:, self.n :] / 2) * self.Y
        )
        self.strings = ry @ rx

    def fidelity(self):
        self.generate()
        self.strings

        A = tf.linalg.einsum(
            "a...k,b...k -> ab...",
            self.strings[:, :, :, 0],
            tf.math.conj(self.strings[:, :, :, 0]),
        )
        A = tf.abs(tf.math.reduce_prod(A, axis=2)) ** 2

        fid = tf.math.reduce_sum(A) / self.N**2

        # fid = tf.linalg.set_diag(fid, tf.zeros_like(fid[:, 0]))
        # fid = tf.math.reduce_max(fid, axis=1)
        # fid = tf.math.reduce_sum(fid) / self.N

        return fid

    def generate_circuits(self, grid=False):
        circuit_list = []
        unitary_list = []
        self.generate()
        angles = np.real(self.angles.numpy()[:, :, 0, 0])
        for i in range(self.N):
            q_reg = qk.QuantumRegister(self.n)
            circuit = qk.QuantumCircuit(q_reg)
            for j in range(self.n):
                circuit.rx(angles[i, j], j)
                circuit.ry(angles[i, j + self.n], j)
            circuit_list.append(circuit)
            unitary_list.append(Operator(circuit.reverse_bits()).data)

        unitary_list = tf.convert_to_tensor(unitary_list, dtype=precision)
        return circuit_list, unitary_list

    def optimize(self, steps):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        for i in tqdm(range(steps)):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.parameters)

                loss = self.fidelity()
                grads = tape.gradient(loss, self.parameter_list)
                optimizer.apply_gradients(zip(grads, self.parameter_list))
            print(loss)


class ExecuteAndCollect:
    def setup_circuits(
        self, circuit_target_list=None, N_map=None, N_spam=None, initial_layout=None
    ):
        self.circuit_target_list = circuit_target_list

        self.n = len(circuit_target_list[0].qregs[0])
        self.data_list = []

        for circuit_target in self.circuit_target_list:
            inputs_map, circuit_list_map = generate_pauli_circuits(
                self.n, circuit_target, N=N_map
            )
            inputs_spam, circuit_list_spam = generate_pauliInput_circuits(self.n)

            self.data_list.append(
                [inputs_map, circuit_list_map, inputs_spam, circuit_list_spam]
            )

        if initial_layout is None:
            self.initial_layout = list(range(self.n))
        else:
            self.initial_layout = initial_layout

    def execute_circuits(
        self,
        backend,
        shots_map,
        shots_spam,
        filename=None,
        concatenate=False,
    ):
        self.result_list = []
        self.shots_map = shots_map
        self.shots_spam = shots_spam

        self.result_list = []

        for i, data in enumerate(self.data_list):
            inputs_map, circuit_list_map, inputs_spam, circuit_list_spam = data
            if concatenate:
                circuit_list = circuit_list_spam + circuit_list_map
                counts_list = self.runner(
                    circuit_list, backend, shots=shots_map, filename=filename
                )
                counts_spam = counts_list[: len(circuit_list_spam)]
                counts_map = counts_list[len(circuit_list_spam) :]
            else:
                counts_map = self.runner(
                    circuit_list_map, backend, shots=shots_map, filename=filename
                )
                counts_spam = self.runner(
                    circuit_list_spam, backend, shots=shots_spam, filename=filename
                )

            probs_map = counts_to_probs(counts_map)
            probs_spam = counts_to_probs(counts_spam)
            self.result_list.append([inputs_map, probs_map, inputs_spam, probs_spam])

            with open("../../data/" + filename + str(i), "wb") as handle:
                pickle.dump(self.result_list[-1], handle)

    def runner(self, circuit_list, backend, shots, filename, scheduling_method="asap"):
        N = len(circuit_list)
        num_batches = (N + 500 - 1) // 500
        circuit_batch_list = [
            circuit_list[500 * i : 500 * (i + 1)] for i in range(num_batches)
        ]
        counts_list = []
        for i, circuit_batch in enumerate(tqdm(circuit_batch_list)):
            num_parcels = (len(circuit_batch) + 100 - 1) // 100
            circuit_parcel_list = [
                circuit_batch[100 * j : 100 * (j + 1)] for j in range(num_parcels)
            ]
            job_list = []

            for circuit_parcel in circuit_parcel_list:
                trans_circ_list = qk.transpile(
                    circuit_parcel,
                    backend,
                    optimization_level=0,
                    seed_transpiler=42,
                    scheduling_method=scheduling_method,
                    initial_layout=self.initial_layout,
                )

                job = backend.run(trans_circ_list, shots=shots)

                job_list.append(job)

            result_list = []
            for job in tqdm(job_list):
                result_list.append(job.result())

            for result, circuit_parcel in zip(result_list, circuit_parcel_list):
                counts_list.extend(
                    [result.get_counts(circuit) for circuit in circuit_parcel]
                )

            probs = counts_to_probs(counts_list[-500:])

            with open("../../data/" + filename + str(i) + "_backup", "wb") as handle:
                pickle.dump(probs, handle)

        return counts_list
