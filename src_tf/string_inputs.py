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
from experimental import *

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


class EntangledPauli:
    def __init__(self, n, N):
        self.n = n
        self.d = 2**n
        self.N = N

    
